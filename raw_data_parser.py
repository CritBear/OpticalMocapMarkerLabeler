import numpy as np
import trc_parser
import os
import pandas as pd

class Parser:
    def __init__(self):
        self.save_name = None
        self.save_path = 'labeling_npy/'

    def set_save_path(self, path):
        self.save_path = path + '/'

    def run(self):
        print('Enter csv files directory path : ', end='')
        self.dir_path = input()

        print('Enter save file name : ', end='')
        self.save_name = input()
        self.save_path = self.save_path + self.save_name

        # # parse trc file ( existing training data )
        # self.parse_trc('../training_data', self.save_path)

        # # parse csv file ( capture data that exported by motive take file )
        # self.parse_csv('labeling_csv/Labeling_Take 2022-01-15 04.53.00 PM_002.csv', self.save_path, header=5)

        # parse all csv files
        self.parse_all_csv(self.dir_path, self.save_path, header=5)
        print('Save completed.')
        print('Saved file path : ' + self.save_path + '.npy')


    def parse_trc(self, data_dir, save_path):
        all_frames = np.empty((1, 19, 3))

        for user_dir in os.listdir(data_dir):
            # if user_dir != 'User3':
            #     continue
            for capture_dir in os.listdir(data_dir + '/' + user_dir):
                for trc_file in os.listdir(data_dir + '/' + user_dir + '/' + capture_dir):
                    if 'right' not in trc_file: # use only right hand
                        continue
                    file_path = data_dir + '/' + user_dir + '/' + capture_dir + '/' + trc_file
                    print(file_path)

                    with open(file_path) as file:
                        trc = trc_parser.TRC(file)

                    # print(all_frames.shape, trc.frames.shape)
                    all_frames = np.vstack([all_frames, trc.frames])
                    # break

                # break
            # break
        all_frames = np.delete(all_frames, [0, 0], axis=0)
        print(all_frames.shape)
        np.save(save_path, all_frames)

    def parse_csv(self, data_dir, save_path, header):

        # Read csv data using pandas
        # In motive data, header is usually 5
        csv_data = pd.read_csv(data_dir, header=header)

        # If any values is empty(nan), drop the row
        csv_data = csv_data.dropna(axis=0)

        # Exclude frame number and delta time
        # Change pandas to numpy
        marker_data = csv_data.values[:, 2:]

        # Change shape : ( 57 ) -> ( 19 x 3 ) markers' xyz position
        marker_data = marker_data.reshape(marker_data.shape[0], -1, 3)

        print('Data shape: ', marker_data.shape)

        np.save(save_path, marker_data)

    def parse_all_csv(self, dir_path, save_path, header):

        # refer parse_csv comment

        dataset = np.zeros((0, 19, 3))
        for filename in os.listdir(dir_path):
            csv_data = pd.read_csv(os.path.join(dir_path, filename), header=header)
            csv_data = csv_data.dropna(axis=0)
            marker_data = csv_data.values[:, 2:]
            marker_data = marker_data.reshape(marker_data.shape[0], -1, 3)
            dataset = np.concatenate((dataset, marker_data), axis=0)
            print('Data shape: ', marker_data.shape, filename)

        print('Dataset shape: ', dataset.shape)

        np.save(save_path, dataset)
