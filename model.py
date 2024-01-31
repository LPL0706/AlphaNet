#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 09:28:01 2022

@author: liupeilin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import pickle

import torch
from torch import nn
from torch.optim import RMSprop, Adam
from torch.utils.data import Dataset, DataLoader

from torchtools import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

class myLoss(nn.Module): # 自定义的适用于股票收益预测的损失函数，包括-IC与AdjMSE
    def __init__(self, mode='IC'):
        super(myLoss, self).__init__()
        self.mode = mode

    def forward(self, pred, label):
        if self.mode == 'IC':
            loss = -self.tensor_corr(pred, label)
        return loss
    
    def tensor_corr(self, x, y): # 计算2个tensor张量的person相关系数
        x, y = x.reshape(-1), y.reshape(-1)
        x_mean, y_mean = torch.mean(x), torch.mean(y)
        corr = (torch.sum((x - x_mean) * (y - y_mean))) / (torch.sqrt(torch.sum((x - x_mean) ** 2)) * torch.sqrt(torch.sum((y - y_mean) ** 2)))
        return corr
    
class mySet(Dataset):    
    def __init__(self, images):
        super(mySet, self).__init__()
        self.data = images
        
    def __getitem__(self, x):
        return self.data[x]
    
    def __len__(self):
        return len(self.data)

def numpy_fill(array):
    where_are_nan = np.isnan(array)
    where_are_inf = np.isinf(array)
    where_are_none = pd.isnull(array)
    array[where_are_nan] = 0
    array[where_are_inf] = 0
    array[where_are_none] = 0
    
def generate(l1):
    if len(l1) == 1:
        return []
    v = [[l1[0], i] for i in l1[1: ]]
    l1 = l1[1: ]
    return v + generate(l1)

class AlphaNet(nn.Module):
    
    def __init__(self, input_channel, fc1_neuron, fc2_neuron, fc3_neuron, fcast_neuron, feat_nums):
        super(AlphaNet, self).__init__()
        self.input_channel = input_channel
        self.fc1_neuron = fc1_neuron
        self.fc2_neuron = fc2_neuron
        self.fc3_neuron = fc3_neuron
        self.fcast_neuron = fcast_neuron
        self.batchnorm = nn.BatchNorm2d(input_channel)
        self.dropout = nn.Dropout(0.5)
        self.conv = nn.Conv2d(1, 16, kernel_size=(1, 3))
        self.fc1 = nn.Linear(self.fc1_neuron, self.fc2_neuron)
        self.fc2 = nn.Linear(self.fc2_neuron, self.fc3_neuron)
        self.fc3 = nn.Linear(self.fc3_neuron, self.fcast_neuron)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        num = generate(list(range(feat_nums)))
        num_rev = [] # 数组反转import torch
        for l in num:
            l1 = l.copy()
            l1.reverse()
            num_rev.append(l1)
        self.num = num
        self.num_rev = num_rev    
        
    def forward(self, data):
        # 运算符
        conv1 = self.ts_corr(data, 10).to(torch.float)       
        conv2 = self.ts_cov(data, 10).to(torch.float)
        conv3 = self.ts_stddev(data, 10).to(torch.float)
        conv4 = self.ts_zscore(data, 10).to(torch.float)
        conv5 = self.ts_return(data, 10).to(torch.float)
        conv6 = self.ts_decaylinear(data, 10).to(torch.float)      
        data_conv = torch.cat([conv1, conv2, conv3, conv4, conv5, conv6], axis=2)
        data_conv = self.batchnorm(data_conv)
        # 卷积层
        data_conv = self.conv(data_conv)
        data_conv = self.relu(data_conv)
        # 特征展平
        # data_fin = data_conv.flatten(start_dim=1)
        data_fin = data_conv.reshape(data_conv.shape[0], -1) 
        # 全连接隐藏层
        ful_connect = self.dropout(self.relu(self.fc1(data_fin)))
        ful_connect = self.dropout(self.sigmoid(self.fc2(ful_connect)))
        output = self.fc3(ful_connect).T[0]       
        return output.to(torch.float)
    
    def ts_cov(self, data, stride):
        '''Caculate the covariance of four-dimension data'''
        '''data:[N,C,H,W],H:feature of stock data picture,W:price length,C:stock numbers
        num:combination pair,reverse of num'''
        num = self.num
        num_rev = self.num_rev
        # 构建步长列表，如果数据长度不能整除，则取剩下长度，如果剩下长度小于5，则与上一步结合一起
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        data_length = data.shape[3]
        conv_feat = len(num)
        if data_length % stride == 0:
            step_list = list(range(0, data_length + stride, stride))
        elif data_length % stride <= 5:
            mod = data_length % stride
            step_list = list(range(0, data_length - stride, stride)) + [data_length]
        else:
            mod = data_length % stride
            step_list = list(range(0, data_length + stride - mod,stride)) + [data_length]
        l = []
        for i in range(len(step_list) - 1):
            start = step_list[i]
            end = step_list[i + 1]
            sub_data1 = data[:,:, num, start:end]
            sub_data2 = data[:,:, num_rev, start:end]
            mean1 = sub_data1.mean(axis=4, keepdims=True)
            mean2 = sub_data2.mean(axis=4, keepdims=True)
            spread1 = sub_data1 - mean1
            spread2 = sub_data2 - mean2
            cov = ((spread1 * spread2).sum(axis=4, keepdims=True) / (sub_data1.shape[4] - 1)).mean(axis=3, keepdims=True)
            numpy_fill(cov)
            l.append(cov)
        l = np.array(l)
        conv_feat = len(num)
        if data.shape[0] != 1:
            result = np.squeeze(l).transpose(1, 2, 0).reshape(-1, 1, conv_feat, len(step_list) - 1)
        else:
            result = np.squeeze(l).reshape(l.shape[0], 1, conv_feat).transpose(1, 2, 0).reshape(-1, 1, conv_feat, len(step_list) - 1)
        numpy_fill(result)
        return torch.tensor(result)
    
    def ts_corr(self, data, stride):
        num = self.num
        num_rev = self.num_rev
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        data_length = data.shape[3]
        if data_length % stride == 0:
            step_list = list(range(0, data_length + stride, stride))
        elif data_length % stride <= 5:
            mod = data_length % stride
            step_list = list(range(0, data_length - stride, stride)) + [data_length]
        else:
            mod = data_length % stride
            step_list = list(range(0, data_length + stride - mod, stride)) + [data_length]
        l = []
        for i in range(len(step_list) - 1):
            start = step_list[i]
            end = step_list[i + 1]
            sub_data1 = data[:, :, num, start:end]
            sub_data2 = data[:, :, num_rev, start:end]
            std1 = sub_data1.std(axis=4, keepdims=True)
            std2 = sub_data2.std(axis=4, keepdims=True)
            std = (std1 * std2).mean(axis=3, keepdims=True)
            numpy_fill(std)
            l.append(std)
        l = np.array(l)
        conv_feat = len(num)  
        if data.shape[0] != 1:
            std = np.squeeze(l).transpose(1, 2, 0).reshape(-1, 1, conv_feat, len(step_list) - 1)
        else:
            std = np.squeeze(l).reshape(l.shape[0], 1, conv_feat).transpose(1, 2, 0).reshape(-1, 1, conv_feat, len(step_list) - 1)
        cov = self.ts_cov(data, stride)
        fct = (sub_data1.shape[4] - 1) / sub_data1.shape[4]
        result = np.array((cov / torch.from_numpy(std)) * fct)
        numpy_fill(result)
        return torch.tensor(result)
    
    def ts_stddev(self, data, stride):
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        data_length = data.shape[3]
        feat_num = data.shape[2]
        if data_length % stride == 0:
            step_list = list(range(0, data_length + stride, stride))
        elif data_length % stride <= 5:
            mod = data_length % stride
            step_list = list(range(0, data_length - stride, stride)) + [data_length]
        else:
            mod = data_length % stride
            step_list = list(range(0, data_length + stride - mod, stride)) + [data_length]
        l = []
        for i in range(len(step_list) - 1):
            start = step_list[i]
            end = step_list[i + 1]
            sub_data1 = data[:,:,:, start:end]
            std1 = sub_data1.std(axis=3, keepdims=True)
            numpy_fill(std1)
            l.append(std1)
        l = np.array(l)
        if data.shape[0] != 1:
            std = np.squeeze(l).transpose(1, 2, 0).reshape(-1, 1, feat_num, len(step_list) - 1)
        else:
            std = np.squeeze(l).reshape(l.shape[0], 1, feat_num).transpose(1, 2, 0).reshape(-1, 1, feat_num, len(step_list) - 1)
        numpy_fill(std)
        return torch.tensor(std)
    
    def ts_zscore(self, data, stride):
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        data_length = data.shape[3]
        feat_num = data.shape[2]
        if data_length % stride == 0:
            step_list = list(range(0, data_length + stride, stride))
        elif data_length % stride <= 5:
            mod = data_length % stride
            step_list = list(range(0, data_length - stride, stride)) + [data_length]
        else:
            mod = data_length % stride
            step_list = list(range(0, data_length + stride-mod, stride)) + [data_length]
        l = []
        for i in range(len(step_list) - 1):
            start = step_list[i]
            end = step_list[i + 1]
            sub_data1 = data[:,:,:, start: end]
            mean = sub_data1.mean(axis=3, keepdims=True)
            std = sub_data1.std(axis=3, keepdims=True)
            z_score = mean / std
            numpy_fill(z_score)
            l.append(z_score)
        l = np.array(l)
        if data.shape[0] != 1:
            z_score = np.squeeze(l).transpose(1, 2, 0).reshape(-1, 1, feat_num, len(step_list) - 1)
        else:
            z_score = np.squeeze(l).reshape(l.shape[0], 1, feat_num).transpose(1, 2, 0).reshape(-1, 1, feat_num, len(step_list) - 1)
        numpy_fill(z_score)
        return torch.tensor(z_score)
    
    def ts_return(self, data, stride):
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        data_length = data.shape[3]
        feat_num = data.shape[2]
        if data_length % stride == 0:
            step_list = list(range(0, data_length + stride, stride))
        elif data_length % stride <= 5:
            mod = data_length % stride
            step_list = list(range(0, data_length - stride, stride)) + [data_length]
        else:
            mod = data_length % stride
            step_list = list(range(0, data_length + stride - mod, stride)) + [data_length]
        l = []
        for i in range(len(step_list) - 1):
            start = step_list[i]
            end = step_list[i + 1]
            sub_data1 = data[:,:,:, start:end]
            ret = sub_data1[:,:,:, -1] / sub_data1[:,:,:, 0] - 1 
            numpy_fill(ret)               
            l.append(ret)
        l = np.array(l)
        if data.shape[0] != 1:
            return_ = np.squeeze(l).transpose(1, 2, 0).reshape(-1, 1, feat_num, len(step_list) - 1)
        else:
            return_ = np.squeeze(l).reshape(l.shape[0], 1, feat_num).transpose(1, 2, 0).reshape(-1, 1, feat_num, len(step_list) - 1)
        numpy_fill(return_)
        return torch.tensor(return_)
    
    def ts_decaylinear(self, data, stride):
        if len(data.shape) != 4:
            raise Exception('Input data dimensions should be [N,C,H,W]')
        data_length = data.shape[3]
        feat_num = data.shape[2]
        if data_length % stride == 0:
            step_list = list(range(0, data_length + stride, stride))
        elif data_length % stride <= 5:
            mod = data_length % stride
            step_list = list(range(0, data_length - stride, stride)) + [data_length]
        else:
            mod = data_length % stride
            step_list = list(range(0, data_length + stride - mod, stride)) + [data_length]
        l = []
        for i in range(len(step_list) - 1):
            start = step_list[i]
            end = step_list[i + 1]
            time_spread = end - start
            weight = np.arange(1, time_spread + 1)
            weight = weight / (weight.sum())
            sub_data1 = (data[:,:,:, start: end] * weight).sum(axis=3, keepdims=True)
            numpy_fill(sub_data1)
            l.append(sub_data1)
        l = np.array(l)
        if data.shape[0] != 1:
            decay = np.squeeze(l).transpose(1, 2, 0).reshape(-1, 1, feat_num, len(step_list) - 1)
        else:
            decay = np.squeeze(l).reshape(l.shape[0], 1, feat_num).transpose(1, 2, 0).reshape(-1, 1, feat_num, len(step_list) - 1)
        numpy_fill(decay)
        return torch.tensor(decay)
    
@torch.no_grad()
def validset_loss(model, valid_days): # 给出模型在验证集上的整体损失
    model.eval() # 固定模型参数
    loss_func = nn.MSELoss()
    # loss_func = myLoss(mode='IC')
    valid_loss = 0
    for date in valid_days:
        data = pd.read_pickle('/Users/liupeilin/Desktop/AlphaNet/Haitong_data/' + date + '.pkl')
        feature = list(data['X'])
        label = list(data['Y2'])
        valid_data = []
        for i in range(len(feature)):
            X = feature[i]
            valid_data.append((X, label[i]))
        valid_set = mySet(valid_data)
        valid_loader = DataLoader(valid_set, batch_size=1000, shuffle=False)
        
        for i, data in enumerate(valid_loader):
            X, y = data
            y_pred = model(X.detach().numpy())
            loss = loss_func(y_pred, y.to(torch.float))
            if pd.isnull(loss.item()):
                continue
            valid_loss += loss.item()
    valid_loss = valid_loss / len(valid_days)
    print('average loss on validset: ', valid_loss)
    return valid_loss

@torch.no_grad()
def model_backtest(alphanet, all_stock_pool, train_days, test_days): # 给出模型在训练集和测试集上的IC和RankIC，同时记录测试集上的alpha表
    alphanet.eval() # 固定模型参数，开始测试模式
    train_IC_list = []
    train_rank_IC_list = []
    test_IC_list = []
    test_rank_IC_list = []
    alpha = all_stock_pool.copy()   
    for date in train_days + test_days:
        data = pd.read_pickle('/Users/liupeilin/Desktop/AlphaNet/Haitong_data/' + date + '.pkl')
        feature = list(data['X'])
        label = list(data['Y2'])
        stock_pool = list(data['Uid'])

        data = []
        for i in range(len(feature)):
            data.append((feature[i], label[i]))
        data_set = mySet(data)
        data_loader = DataLoader(data_set, batch_size=len(data_set), shuffle=False)

        y_pred = []
        for i, data in enumerate(data_loader):
            X, y = data
            pred = alphanet(X.detach().numpy())
            for j in range(len(pred)):
                y_pred.append(pred[j].item())
    
        # 给出每一个交易日的IC和rank_IC
        y = pd.Series(y)
        y_rank = y.rank()
        y_pred = pd.Series(y_pred)
        y_pred_rank = y_pred.rank()
        IC = y_pred.corr(y)
        RankIC = y_pred_rank.corr(y_rank)
        if date <= train_days[-1]: # 训练集
            train_IC_list.append(IC)
            train_rank_IC_list.append(RankIC)
        else: # 测试集，同时记录alpha表
            test_IC_list.append(IC)
            test_rank_IC_list.append(RankIC)            
            pred_df = pd.DataFrame({'Uid': stock_pool, date: y_pred}, index=range(len(y_pred)))             
            alpha = pd.merge(alpha, pred_df, left_on='Uid', right_on='Uid', how='outer')
            alpha.sort_values('Uid', inplace=True)
    # 把alpha表转化为回测系统要求的格式
    alpha = alpha.set_index('Uid')
    alpha = alpha.T
    alpha = alpha.reset_index()
    alpha = alpha.rename(columns={'index':'Date'})
    alpha.to_csv('/Users/liupeilin/Desktop/Haitong_alpha.csv', index=False)
    return train_IC_list, train_rank_IC_list, test_IC_list, test_rank_IC_list

def model_train(train_days, valid_days, batch_size=1000, max_epoch=10, lr=1e-5, patience=2):
    # input_channel, fc1_neuron, fc2_neuron, fcast_neuron, 特征数
    alphanet = AlphaNet(1, 4864, 256, 30, 1, 16)
    # alphanet = torch.load('/Users/liupeilin/Desktop/model_v2_MSE.pkl')
    loss_func = nn.MSELoss()
    # loss_func = myLoss(mode='IC')
    optimizer = Adam(alphanet.parameters(), lr=lr)
    
    path = '/Users/liupeilin/Desktop/'
    name = 'Haitong_model'
    early_stopping = EarlyStopping(path=path, name=name, patience=patience)
    
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(max_epoch):
        epoch_loss = 0
        for date in train_days:
            day_loss = 0
            data = pd.read_pickle('/Users/liupeilin/Desktop/AlphaNet/Haitong_data/' + date + '.pkl')
            feature = list(data['X'])
            label = list(data['Y2'])

            train_data = []
            for i in range(len(feature)):
                train_data.append((feature[i], label[i]))

            train_set = mySet(train_data)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
            
            for i, data in enumerate(train_loader):
                optimizer.zero_grad() # 梯度清零
                X, y = data
                y_pred = alphanet(X.detach().numpy())
                loss = loss_func(y_pred, y.to(torch.float))
                if pd.isnull(loss.item()):
                    continue
                loss.backward() # 反向传播
                optimizer.step()
                day_loss += loss.item()
            # print("Epoch %d, Date %s loss: %f"%(epoch + 1, date, day_loss))
            epoch_loss += day_loss
        epoch_loss /= len(train_days)
        train_loss_list.append(epoch_loss)
        valid_loss = validset_loss(alphanet, valid_days)
        valid_loss_list.append(valid_loss)
        print("\n##### Epoch %d average loss: "%int(epoch + 1), epoch_loss, '#####\n')
        '''
        # 记录迭代过程中模型在训练集和验证集上的损失
        with open('训练日志.txt', 'a') as f:
            f.write('epoch_' + str(epoch + 1) + ', ' + str(epoch_loss) + ', ' + str(valid_loss) + '\n')
            f.close()
        '''
        # 早停，可以设置至少训练n个epoch，避免因前期验证集损失波动导致训练中止
        if epoch >= 4:
            early_stopping(valid_loss, alphanet)
            if early_stopping.early_stop:
                print("Early stopping")
                '''
                with open('训练日志.txt', 'a') as f:
                    f.write('EarlyStopping\n')
                    now_time = datetime.datetime.now()
                    f.write(str(now_time))
                    f.close()
                '''
                break
    
    plt.subplot(121)
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.subplot(122)
    plt.plot(range(len(valid_loss_list)), valid_loss_list)
    
    # torch.save(alphanet, '/Users/liupeilin/Desktop/AlphaNet/MSE_model/model.pkl') # 保存模型
    return alphanet, train_loss_list, valid_loss_list

def main():
    tradingdays = pd.read_csv('/Users/liupeilin/Desktop/股票数据/tradingdays.csv')
    # tradingdays = tradingdays[(tradingdays['Date'] >= '2015-02-17') & (tradingdays['Date'] <= '2020-12-17')]
    tradingdays = tradingdays[(tradingdays['Date'] >= '2015-02-25') & (tradingdays['Date'] <= '2020-03-28')]
    tradingdays = list(tradingdays['Date'])

    train_days = tradingdays[0:800]
    valid_days = tradingdays[800:1000]
    test_days = tradingdays[1000:1200]
    
    all_stock_pool = pd.read_csv('/Users/liupeilin/Desktop/股票数据/all_stock_pool.csv')
    
    # model = torch.load('/Users/liupeilin/Desktop/model_v2_MSE.pkl')
    time1 = time()
    # 模型训练
    model, train_loss_list, valid_loss_list = model_train(train_days, valid_days, max_epoch=150, lr=1e-6, patience=2)
    time2 = time()
    print('训练用时: ', time2 - time1, 's')
    # 模型在训练集上的表现，用于判断数据是否得到了有效的训练
    train_IC_list, train_rank_IC_list, test_IC_list, test_rank_IC_list = model_backtest(model, all_stock_pool, train_days, test_days)
    train_IC_avg = np.mean(train_IC_list)
    train_IC_std = np.std(train_IC_list)
    train_IR = train_IC_avg / train_IC_std 
    train_rank_IC_avg = np.mean(train_rank_IC_list)
    train_win_rate = len([i for i in train_IC_list if i > 0]) / len(train_IC_list)
    print('训练集: 均值IC =', train_IC_avg)
    print('训练集: IR =', train_IR)
    print('训练集: 均值RankIC =', train_rank_IC_avg)
    print('训练集: 胜率 =', train_win_rate)
    
    test_IC_avg = np.mean(test_IC_list)
    test_rank_IC_avg = np.mean(test_rank_IC_list)
    test_IC_std = np.std(test_IC_list)
    test_IR = test_IC_avg / test_IC_std 
    test_win_rate = len([i for i in test_IC_list if i > 0]) / len(test_IC_list)
    print('测试集: 均值IC =', test_IC_avg)
    print('测试集: IR =', test_IR)
    print('测试集: 均值RankIC =', test_rank_IC_avg)
    print('测试集: 胜率 =', test_win_rate)
    
if __name__ == '__main__':
    main()





