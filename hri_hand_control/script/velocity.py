import time
import pickle 
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class RegNet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden):
        super(RegNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.fc3 = torch.nn.Linear(hidden, hidden)
        self.bn3 = torch.nn.BatchNorm1d(hidden)
        self.fc4 = torch.nn.Linear(hidden, hidden)
        self.bn4 = torch.nn.BatchNorm1d(hidden)
        self.fc5 = torch.nn.Linear(hidden, hidden)
        self.bn5 = torch.nn.BatchNorm1d(hidden)
        self.fc6 = torch.nn.Linear(hidden, 300)
        self.bn6 = torch.nn.BatchNorm1d(300)
        self.fc7 = torch.nn.Linear(300, output_size)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.xavier_normal_(self.fc4.weight)
        torch.nn.init.xavier_normal_(self.fc5.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        #x = self.fc2(x)
        #x = self.bn2(x)
        #x = F.relu(x)
        #x = self.fc3(x)
        #x = self.bn3(x)
        #x = F.relu(x)
        #x = self.fc4(x)
        #x = self.bn4(x)
        #x = F.relu(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.fc7(x)
        # x = torch.sigmoid(x)

        return x

class velocity_predictor:
    def __init__(self, dataset_name, MIN=0, MAX=0.012, velocity=0.001):
        self.fingers_state = {'index_control' : None,
                             'middle_control' : None,
                             'ring_control' : None,
                             'little_control' : None,
                             'thumb_joint_control' : None,
                             'thumb_control' : None
                            }
        self.reset()
                
        self.MIN = MIN
        self.MAX = MAX
        self.velocity = velocity

        with open('/home/naoki/ros_ws_v1/src/myo_ros_v1/' + dataset_name + '/' + 'standard_reg0.pickle', mode='rb') as fp:
            self.ss_reg0 = pickle.load(fp)
        with open('/home/naoki/ros_ws_v1/src/myo_ros_v1/' + dataset_name + '/' + 'standard_reg2.pickle', mode='rb') as fp:
            self.ss_reg2 = pickle.load(fp)

        self.net_reg0 = RegNet(197, 1, 1500)
        self.net_reg0.load_state_dict(torch.load('/home/naoki/ros_ws_v1/src/myo_ros_v1/' + dataset_name + '/' + 'net_reg0.pth'))
        self.net_reg0.train(False)

        self.net_reg2 = RegNet(197, 1, 1500)
        self.net_reg2.load_state_dict(torch.load('/home/naoki/ros_ws_v1/src/myo_ros_v1/' + dataset_name + '/' + 'net_reg2.pth'))
        self.net_reg2.train(False)

        self.feature_tmp = None

    def __call__(self, hand, feature_vector):
        self.feature_tmp = feature_vector
        self._switch(hand)
        return self.fingers_state

    def _switch(self, hand):
        if hand == 0:
            self.all_flex()
        elif hand == 1:
            self.all_ext()
        elif hand == 2:
            self.index_flex()
        else:
            pass

    def reset(self):
        for i, m in self.fingers_state.items():
            self.fingers_state[i] = 0
    
    def all_ext(self):
        for i, m in self.fingers_state.items():
            if self.fingers_state[i] > self.MIN:
                self.fingers_state[i] -= self.velocity
    
    def all_flex(self):
        for i, m in self.fingers_state.items():
            if i != 'thumb_control' and i != 'thumb_joint_control':
                if self.fingers_state[i] < self.MAX:
                    # predict velocity by reg0
                    feature_tmp =self.ss_reg0.transform(self.feature_tmp)
                    output = self.net_reg0(torch.from_numpy(feature_tmp).float())
                    pred_v = output.detach().numpy()[0][0]
                    print(pred_v)
                    pred_v_tmp = pred_v * 0.012 / 300
                    if self.fingers_state[i] + pred_v_tmp > self.MAX:
                        self.fingers_state[i] = self.MAX
                    else:
                        self.fingers_state[i] += pred_v_tmp

                    # self.fingers_state[i] += self.velocity

    def index_flex(self):
        for i, m in self.fingers_state.items():
            if i == 'index_control':
                if self.fingers_state[i] < self.MAX:
                    # predict velocity by reg2
                    feature_tmp =self.ss_reg0.transform(self.feature_tmp)
                    output = self.net_reg0(torch.from_numpy(feature_tmp).float())
                    pred_v = output.detach().numpy()[0][0]
                    print(pred_v)
                    pred_v_tmp = pred_v * 0.012 / 300
                    if self.fingers_state[i] + pred_v_tmp > self.MAX:
                        self.fingers_state[i] = self.MAX
                    else:
                        self.fingers_state[i] += pred_v_tmp
                    # self.fingers_state[i] += self.velocity
            else:
                if self.fingers_state[i] > self.MIN:
                    self.fingers_state[i] -= self.velocity


class control:
    def __init__(self, hj_tf, sleep_time = 0.04):
        self.hj_tf = hj_tf
        self.stm_index_id = 0x01
        self.stm_middle_id = 0x02
        self.stm_ring_id = 0x03
        self.stm_little_id = 0x04
        self.stm_thumb_id = 0x05
        self.stm_thumb_joint_id = 0x06
        self.sleep_time = sleep_time

    def move(self, fingers_state):
        index_pris_val = fingers_state['index_control']
        middle_pris_val = fingers_state['middle_control']
        ring_pris_val = fingers_state['ring_control']
        little_pris_val = fingers_state['little_control']
        thumb_pris_val = fingers_state['thumb_joint_control']
        thumb_joint_pris_val = fingers_state['thumb_control']

        # index
        self.hj_tf.index_control(index_pris_val)
        self.hj_tf.hj_finger_control(self.stm_index_id, index_pris_val)

        # middle
        self.hj_tf.middle_control(middle_pris_val)
        self.hj_tf.hj_finger_control(self.stm_middle_id, middle_pris_val)

        # ring
        self.hj_tf.ring_control(ring_pris_val)
        self.hj_tf.hj_finger_control(self.stm_ring_id, ring_pris_val)

        # little
        self.hj_tf.little_control(little_pris_val)
        self.hj_tf.hj_finger_control(self.stm_little_id, little_pris_val)

        # thumb_joint
        self.hj_tf.thumb_joint_control(thumb_joint_pris_val)
        self.hj_tf.hj_finger_control(self.stm_thumb_joint_id, thumb_joint_pris_val)

        # thumb
        self.hj_tf.thumb_control(thumb_pris_val)
        self.hj_tf.hj_finger_control(self.stm_thumb_id, thumb_pris_val)

        #time.sleep(self.sleep_time)



