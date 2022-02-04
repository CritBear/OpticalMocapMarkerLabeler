import numpy as np
import pandas as pd
import cv2
import time
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Network(nn.Module):
    # VGG style CNN

    def __init__(self):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), padding=0),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), padding=0, stride=(2,2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), padding=0, stride=(2, 2))
        )
        # view 128 x 9 x 9 -> 10368
        self.fc = nn.Sequential(
            nn.Linear(10368, 2048),
            nn.ReLU(),
            nn.Linear(2048, 57) #57
        )
        # view 57 -> 19 x 3

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.view(out.size(0), 19, 3)
        return out

class SpatialImage_Dataset(Dataset):
    def __init__(self, x, y, device):
        #self.x_data = torch.as_tensor(np.array(x).astype(np.float32)).to(device)
        #self.y_data = torch.as_tensor(np.array(y).astype(np.float32)).to(device)
        self.x_data = torch.from_numpy(x).float()
        self.y_data = torch.from_numpy(y).float()

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


    def __len__(self):
        return len(self.x_data)


class Trainer():
    def __init__(self):
        print('Enter img, pts_local file path+name : ', end='')
        self.file_path = input()

        print('Enter save model path : ', end='')
        self.model_path = input()

    def train(self):
        epoch = 100
        batch_size = 128
        init_lr = 0.001

        is_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if is_cuda else 'cpu')

        model = Network().to(device)

        # parts = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 84510]
        # img_all_frames = np.load('img_' + str(parts[0]) + '_' + str(parts[1]) + '.npy')
        # pts_local_all_frames = np.load('pts_local_' + str(parts[0]) + '_' + str(parts[1]) + '.npy')
        # for i in range(1, len(parts) - 1):
        #     img_temp = np.load('img_' + str(parts[i]) + '_' + str(parts[i + 1]) + '.npy')
        #     pts_local_temp = np.load('pts_local_' + str(parts[i]) + '_' + str(parts[i + 1]) + '.npy')
        #     img_all_frames = np.concatenate((img_all_frames, img_temp), axis=0)
        #     pts_local_all_frames = np.concatenate((pts_local_all_frames, pts_local_temp), axis=0)

        # img : input data
        # pts_local : label data
        img_all_frames = np.load(self.file_path + '_img.npy')
        pts_local_all_frames = np.load(self.file_path + '_pts_local.npy')

        print('img_all_frames shape: ', img_all_frames.shape)
        print('pts_local_all_frames shape: ', pts_local_all_frames.shape)

        img_training = img_all_frames
        pts_local_training = pts_local_all_frames

        print('img_training shape: ', img_training.shape)
        print('pts_local_training shape: ', pts_local_training.shape)

        train_input = img_training.reshape((img_training.shape[0], 1, 52, 52))
        train_label = pts_local_training.reshape((pts_local_training.shape[0], 19, 3))
        print('train input shape: ', train_input.shape)
        print('train label shape: ', train_label.shape)

        train_dataset = SpatialImage_Dataset(train_input, train_label, device=device)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        criterion = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

        print("######### Start Train #########")
        train_start_time = time.time()

        # Visualization result loss
        epoch_list = range(epoch)
        train_loss_list = []

        for epoch_idx in range(epoch):
            epoch_start_time = time.time()
            train_loss = 0.

            for batch_idx, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # print(inputs.shape)
                outputs = model(inputs)

                # print(outputs.shape)
                # print(labels.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= (batch_idx + 1)
            elapsed_time = time.time() - epoch_start_time
            print('%5d | Train Loss: %.7f | time: %.3f' % (epoch_idx + 1, train_loss, elapsed_time))
            train_loss_list.append(train_loss)

            if epoch_idx == epoch - 1:
                torch.save(model.state_dict(), self.model_path + '.pt')
                print('Model saved.')

        print('Entire time: %.3f' % (time.time() - train_start_time))

        plt.plot(epoch_list, train_loss_list)
        plt.show()

    def save_onnx(self):
        # save network model for cross platform
        model_path = self.model_path + '.pt'

        is_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if is_cuda else 'cpu')
        model = Network().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        img = np.load(self.file_path + '_img.npy')

        network_input = img[0].reshape(1, 1, 52, 52)
        network_input = torch.from_numpy(network_input).float().to(device)

        network_output = model(network_input)

        torch.onnx.export(model, network_input, self.model_path + '.onnx', verbose=True, example_outputs=network_output)

        network_output = network_output.to('cpu')
        network_output = network_output.detach().numpy().reshape(-1, 3)

        np.save('test_predict', network_output)

    def statistic_backhand_marker(self, data):
        # data must be labeled
        # index 16 -> Backhand 1
        # index 17 -> Backhand 2
        # index 18 -> Backhand 3

        # backhand_data = data[:, :3, :]
        # vector_12 = backhand_data[:, 1, :] - backhand_data[:, 0, :]
        # vector_13 = backhand_data[:, 2, :] - backhand_data[:, 0, :]
        # vector_23 = backhand_data[:, 2, :] - backhand_data[:, 1, :]

        vector_12 = data[:, 17, :] - data[:, 16, :]
        vector_13 = data[:, 18, :] - data[:, 16, :]
        vector_23 = data[:, 18, :] - data[:, 17, :]

        dist_12 = np.linalg.norm(vector_12, axis=1)
        dist_13 = np.linalg.norm(vector_13, axis=1)
        dist_23 = np.linalg.norm(vector_23, axis=1)

        mean_12 = np.mean(dist_12)
        max_12 = np.max(dist_12)
        min_12 = np.min(dist_12)

        mean_13 = np.mean(dist_13)
        max_13 = np.max(dist_13)
        min_13 = np.min(dist_13)

        mean_23 = np.mean(dist_23)
        max_23 = np.max(dist_23)
        min_23 = np.min(dist_23)

        print('1 to 2 : ', mean_12, max_12, min_12)
        print('1 to 3 : ', mean_13, max_13, min_13)
        print('2 to 3 : ', mean_23, max_23, min_23)

        return mean_12, mean_13, mean_23


