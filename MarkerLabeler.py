import raw_data_parser
import parse_data_viewer
import label_generator
import labeling_trainer


class MarkerLabeler:
    def __init__(self):
        pass

    def run(self):
        while True:
            print('1 : Parse raw data')
            print('1-1 : View parsed data')
            print('2 : Generate training data')
            print('3 : Training neural network')

            print('Enter command : ', end='')
            command = input()

            if command == '1':
                self.parse_raw_data()
            elif command == '1-1':
                self.view_parsed_data()
            elif command == '2':
                self.generate_training_data()
            elif command == '3':
                self.training()

    def parse_raw_data(self):
        # Parser
        #   file to read : .csv raw motion data
        #   file to save : .npy shape : (number of frames x 19 x 3)
        #
        #   set_save_path
        #       - Set parsed data
        #       - Example : 'data/parsed_data'
        #       - The directory must be exist
        #   run
        #       - Read all csv file in entered directory path
        #         and save into numpy file
        #       - The config(parameter) is set as motive exported data
        #         if want to change, see raw_data_parser.py

        parser = raw_data_parser.Parser()
        parser.set_save_path('labeling_npy')
        parser.run()

    def view_parsed_data(self):
        # Viewer
        #   file to read : .npy parsed motion data
        #
        #   - view parsed file using vpython(3d render)
        #   - marker color
        #       - red : 17th marker (backhand 1)
        #       - green : 18th marker (backhand 2)
        #       - blue : 19th marker (backhand 3)

        viewer = parse_data_viewer.Viewer()
        viewer.run()

    def generate_training_data(self):
        # LabelGenerator
        #   file to read : .npy parsed motion data
        #   files to save : 1 .npy projected img file
        #                   2 .npy local marker position (frames x 19 x 3)
        #
        #   - Generate training data by projection parsed data
        #   - Projection direction is perpendicular to the palm(center of backhand markers)

        generator = label_generator.LabelGenerator()
        generator.run()

    def training(self):
        # Trainer
        #   files to read : 1 .npy projected img file
        #                   2 .npy local marker position (frames x 19 x 3)
        #   files to save : 1 .pt saved model (pytorch)
        #                   2 .onnx saved model (cross platform)

        trainer = labeling_trainer.Trainer()
        trainer.train()
        trainer.save_onnx()



def main():
    app = MarkerLabeler()
    app.run()



if __name__ == '__main__':
    main()