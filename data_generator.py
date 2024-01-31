#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:08:59 2022

@author: liupeilin
"""
from scipy import io
import pandas as pd
import numpy as np
import math

import warnings
warnings.filterwarnings("ignore")

def data_check(image, earnrate1, earnrate2, earnrate3):
    image_sum = np.sum(image)
    if pd.isnull(earnrate1):
        return False
    if pd.isna(earnrate1):
        return False
    if math.isinf(earnrate1):
        return False
    
    if pd.isnull(earnrate2):
        return False    
    if pd.isna(earnrate2):
        return False 
    if math.isinf(earnrate2):
        return False
    
    if pd.isnull(earnrate3):
        return False    
    if pd.isna(earnrate3):
        return False 
    if math.isinf(earnrate3):
        return False
    
    if pd.isnull(image_sum):
        return False
    if math.isinf(image_sum):
        return False
    if math.isnan(image_sum):
        return False
    if (image != image).any():
        return False

    return True

# 获取股票量价数据
open_filename = '/Users/liupeilin/Desktop/股票数据/adjopen.csv'
high_filename = '/Users/liupeilin/Desktop/股票数据/adjhigh.csv'
low_filename = '/Users/liupeilin/Desktop/股票数据/adjlow.csv'
close_filename = '/Users/liupeilin/Desktop/股票数据/adjclose.csv'
vwap_filename = '/Users/liupeilin/Desktop/股票数据/adjvwap.csv'
volume_filename = '/Users/liupeilin/Desktop/股票数据/adjvolume.csv'
total_filename = '/Users/liupeilin/Desktop/股票数据/totalshare.csv'
float_filename = '/Users/liupeilin/Desktop/股票数据/floatshare.csv'

open_ = pd.read_csv(open_filename).set_index('Date')
high = pd.read_csv(high_filename).set_index('Date')
low = pd.read_csv(low_filename).set_index('Date')
close = pd.read_csv(close_filename).set_index('Date')
vwap = pd.read_csv(vwap_filename).set_index('Date')
volume = pd.read_csv(volume_filename).set_index('Date')
totalshare = pd.read_csv(total_filename).set_index('Date')
freeshare = pd.read_csv(float_filename).set_index('Date')

return1 = close.pct_change() # 复权日收益率
turn = volume / totalshare # 总股本换手率
free_turn = volume / freeshare # 自由流通股换手率
mktcap = close * totalshare # 总市值

tradingdays_list = list(pd.read_csv('/Users/liupeilin/Desktop/股票数据/tradingdays.csv')['Date'])

def data_prepare(code, date, window=30):
    
    index = tradingdays_list.index(date)
    startday = tradingdays_list[index - window]
    endday = tradingdays_list[index - 1]
    buy_day = date
    sell_day = tradingdays_list[index + 1]
    sell_day_2 = tradingdays_list[index + 5]
    sell_day_3 = tradingdays_list[index + 10]
    # 计算收益率
    buy_price = close[code][buy_day: buy_day][buy_day]
    sell_price = close[code][sell_day: sell_day][sell_day]
    sell_price_2 = close[code][sell_day_2: sell_day_2][sell_day_2]
    sell_price_3 = close[code][sell_day_3: sell_day_3][sell_day_3]
    earnrate = sell_price / buy_price - 1
    earnrate_2 = sell_price_2 / buy_price - 1
    earnrate_3 = sell_price_3 / buy_price - 1

    image = []
    # 技术面数据
    image.append(list(open_[code][startday: endday]))
    image.append(list(high[code][startday: endday]))
    image.append(list(low[code][startday: endday]))
    image.append(list(close[code][startday: endday]))
    image.append(list(vwap[code][startday: endday])) 
    image.append(list(volume[code][startday: endday]))
    image.append(list(return1[code][startday: endday]))
    image.append(list(turn[code][startday: endday]))
    image.append(list(free_turn[code][startday: endday]))

    image = np.array([image]).astype('float32')

    return image, earnrate, earnrate_2, earnrate_3

image, earnrate, earnrate_2, earnrate_3 = data_prepare('000001-SZ', '2015-02-16')

all_stock_pool = list(pd.read_csv('/Users/liupeilin/Desktop/股票数据/all_stock_pool.csv')['Uid'])

def alldata(tradingday):   
    image_list = []
    earnrate1_list = []
    earnrate2_list = []
    earnrate3_list = []
    code_list = []
    for i in list(all_stock_pool):
        image, earnrate1, earnrate2, earnrate3 = data_prepare(i, tradingday)
        # 尽量排除一切异常值
        if data_check(image, earnrate1, earnrate2, earnrate3):
            image_list.append(image)
            earnrate1_list.append(earnrate1)
            earnrate2_list.append(earnrate2)
            earnrate3_list.append(earnrate3)
            code_list.append(i)
        else:
            print(data_prepare(i, tradingday))
            print(i)
            # break
    data = pd.DataFrame({'Uid': code_list,
                         'X': image_list,
                         'Y1': earnrate1_list,
                         'Y2': earnrate2_list,
                         'Y3': earnrate3_list})
    return data


for date in tradingdays_list:
    if date >= '2015-02-17' and date <= '2015-02-17':
        data = alldata(date)     
        # data.to_pickle('./data/' + date + '.pkl')
        data.to_pickle('/Users/liupeilin/Desktop/x.pkl')
        print(date, 'is finished')