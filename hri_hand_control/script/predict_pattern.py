import numpy as np
import pywt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import itertools
import pickle

class MyNet_class(nn.Module):
    def __init__(self, input_size, output_size, hidden):
        super(MyNet_class, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.fc4 = nn.Linear(hidden, 300)
        self.bn4 = nn.BatchNorm1d(300)
        self.fc5 = nn.Linear(300, output_size)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.xavier_normal_(self.fc5.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc5(x)
        # x = torch.sigmoid(x)

        return x


class predict_pattern_int():
    def __init__(self, N, dataset_name, ch_list=[0,1,2,5,6], net_tmp_flag=False):
        self.N = N
        
        self.net_tmp_flag = net_tmp_flag
        if self.net_tmp_flag:
            with open('/home/naoki/ros_ws_v1/src/myo_ros_v1/' + dataset_name + '/' + 'standard_pr_tmp.pickle', mode='rb') as fp:
                self.ss_tmp = pickle.load(fp)
            self.net_tmp = MyNet_class(185, 3, 1000)
            self.net_tmp.load_state_dict(torch.load('/home/naoki/ros_ws_v1/src/myo_ros_v1/' + dataset_name + '/' + 'net_pr_tmp.pth'))
            self.net_tmp.train(False)
        else:
            with open('/home/naoki/ros_ws_v1/src/myo_ros_v1/' + dataset_name + '/' + 'standard_pr.pickle', mode='rb') as fp:
                self.ss = pickle.load(fp)
            output_size = 3
            self.net = MyNet_class(197, output_size, 2000)
            self.net.load_state_dict(torch.load('/home/naoki/ros_ws_v1/src/myo_ros_v1/' + dataset_name + '/' + 'net_pr.pth'))
            self.net.train(False)


        self.max_level = pywt.dwt_max_level(self.N, 'db2')
        
        self.ch_list = ch_list
        

    def predict(self, emg, acc, gyro):#emg: (N, 8), acc(gyro): (N/2, 3)
        raw_emg = np.array(emg)[:self.N, self.ch_list].T # raw_emg: (len(ch_list), N)
        # print(emg.shape)
        dwt_emg = self.dwt_vector(raw_emg).flatten()
        processed_imu = self.process_imu(np.concatenate([acc, gyro], axis=1)) # imu: (N/2, 6)
        #print(dwt_emg.shape, processed_imu.shape)
        processed_data = np.concatenate([dwt_emg, processed_imu]).reshape(1,-1)
        
        # print(processed_emg.shape)
        if self.net_tmp_flag:
            processed_data_ss = self.ss_tmp.transform(processed_data[:, :185])
            output = self.net_tmp(torch.from_numpy(processed_data_ss).float())
            pred_int = output.argmax(dim=1, keepdim=True).detach().numpy()[0][0]
            '''
            if pred_int == 0:
                pass
            else:
                pred_int -= 1'''
        else:
            processed_data_ss = self.ss.transform(processed_data)
            output = self.net(torch.from_numpy(processed_data_ss).float())
            pred_int = output.argmax(dim=1, keepdim=True).detach().numpy()[0][0]

        print(pred_int)
        return pred_int, processed_data
    
    def process_imu(self, data):
        mav = np.mean(np.abs(data), axis=0)
        pos = data[1:, :]
        pre = data[:data.shape[0]-1, :]
        wl = np.sum(np.abs(pos - pre), axis=0)
        return np.concatenate([mav, wl])

    
    def dwt_vector(self, data):        #pywt db2 data:(8, N)
        dwt_emg = []
        for _d in data:
            dwt_emg.append(np.array(list(itertools.chain.from_iterable(pywt.wavedec(_d, 'db2', level=self.max_level)))))
        return np.array(dwt_emg)

