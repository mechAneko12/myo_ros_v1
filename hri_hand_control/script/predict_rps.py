import numpy as np
import json
import copy
import scipy.stats as stats

from dwt_process import DWT_N

class MLE:
    #maximum likelihood estimation
    def __init__(self, method='linear', label=None, mu=None, invS=None, S=None):
        self.methods = ['linear', 'mahalanobis']
        if not method in self.methods:
            print("Classification method is invalid.")
            sys.exit(1)
        self.mtd = self.methods.index(method)
        self.label = label
        self.mu = mu
        self.invS =invS
        self.S = S

    def fit(self, x, y):
        self.label = []
        x = np.array(x)
        x_shape = x.shape[1]
        self.mu = np.empty((0, x_shape), float)
        S = np.empty((0, x_shape, x_shape), float)
        self.label = np.unique(y)
        for c in self.label:
            ind = np.where(y == c)[0]
            x_cl = x[ind]
            self.mu = np.append(self.mu, [np.mean(x_cl, axis=0)], axis=0)
            S = np.append(S, [np.cov(x_cl.T)], axis=0)
        if self.mtd == 0:
            self.invS = np.linalg.inv(np.mean(S, axis=0) + 0.000001 * np.identity(x_shape))
        else:
            self.invS = np.empty((0, x_shape, x_shape), float)
            for s in S:
                self.invS = np.append(self.invS, [np.linalg.inv(s + 0.000001 * np.identity(x_shape))], axis=0)
            self.S = S
        return self
    
    def predict(self, x):
        x = np.array(x)
        p = np.empty((0, x.shape[0]), float)
        if self.mtd == 0:
            for l in range(len(self.label)):
                p = np.append(p, self.mu[l][None, :].dot(self.invS).dot(x.T) - self.mu[l][None, :].dot(self.invS).dot(self.mu[l][:, None]) / 2, axis=0)
        else:
            for l in range(len(self.label)):
                # print( self.mu[l][:,None].shape)
                p = np.append(p, [-0.5 * (np.sum((np.dot((x.T - self.mu[l][:, None]).T, self.invS[l]) * ((x.T - self.mu[l][:, None]).T)), axis=1) + np.log(np.linalg.det(self.S[l])))], axis=0)
        p_max = np.argmax(p, axis=0)
        return self.label[p_max]

class predict_rps_int():
    def __init__(self, N, dataset_name):
        self.N = N
        f = open('/home/naoki/hard_ws/src/HRI-Hand-ROS/' + dataset_name+'/mle_data.json', 'r')
        j = json.load(f)
        f.close()
        self.mle = MLE(method=j['method'], label=np.array(j['label']), mu=np.array(j['mu']), invS=np.array(j['invS']), S=np.array(j['S']))
        self.ch_list = [i for i, x in enumerate(j['ch']) if x == 1]
        
        self.dwt = DWT_N(N)
        self.dwt.filter()

    def predict(self, emg):#emg: (32+n_sample-1, N)
        emg = np.array(emg)[:, self.ch_list].T
        # print(emg.shape)
        raw_emg = self.split_emg(emg)
        # print(raw_emg.shape)
        for i in raw_emg:
            for row in i:
                self.dwt.main(row)
        processed_emg = raw_emg[:, :, 2:].reshape(raw_emg.shape[0], -1)
        # print(processed_emg.shape)
        predicted_datas = self.mle.predict(processed_emg)
        return stats.mode(predicted_datas)[0][0]

    def split_emg(self, emg):
        n_sample = emg.shape[1] - self.N + 1
        return_emg = np.empty((0, emg.shape[0], self.N), float)
        for k in range(n_sample):
            return_emg = np.append(return_emg, [emg[:, k:self.N + k]], axis=0)
        return return_emg
