"""
Created on Sun Mar 3 14:23:13 2024
@author: CZP
"""
# import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def read_data(path):
    """
    :param path: 路径
    :return: 返回的df_list
    """
    names = os.listdir(path)
    df_list = [pd.read_csv(os.path.join(path, name)) for name in names]
    i = 0
    for df in df_list:
        if df["SPEED"].iloc[0] != 0:  # 某些开始的值确确实实不为0
            print(names[i])
        i += 1
    df = pd.concat(df_list, axis=0).reset_index()
    used_df = df[['HEIGHT', 'SPEED', 'LATITUDE N/S', 'LONGITUDE E/W']]
    return used_df


def discrete_and_deal_data(df):
    """
    :param df: 总的df
    :return: 处理后的df
    """
    df['acceleration'] = df['SPEED'].diff(1) / 3.6  # 转化为m/s2
    df_index = df['acceleration'].copy()[(df['acceleration']<0) & (df['acceleration']>-0.1)].index
    df['acceleration'][df_index] = 0
    df['acceleration'] = round(df['acceleration'], 1 )
    df['SPEED'] = round(df['SPEED'], 0)
    df.fillna(0.0, inplace=True)  # 下一步，将其转化为speed, acceleration矩阵
    df = df.apply(lambda row: row if row['SPEED'] != 0 else row.update({'acceleration': 0}) or row, axis=1)
    Frequency_df = df.groupby(['acceleration',"SPEED"]).size().reset_index(name='Frequency')
    Frequency_df["Frequency"] = Frequency_df["Frequency"]/df.shape[0]
    return df,Frequency_df


def generate_tpm(df,Frequency_df):
    """
    :param df: 原始的df
    :param Frequency_df: 频率的df
    :return: 返回一个tpm_list
    """
    tpm_list = []
    from tqdm import tqdm
    for i in tqdm(Frequency_df.iterrows()):  # 这样勉强可以接受
        try:
            speed = i[1]["SPEED"]
            a = i[1]["acceleration"]
            intermediate_df = df[(df["SPEED"] == speed) & (df['acceleration'] == a)]
            index = intermediate_df.index + 1
            intermediate_df = df.iloc[index]
            tpm_list.append(intermediate_df.groupby(["SPEED", "acceleration"]).size() / intermediate_df.shape[0])
        except:
            intermediate_index = [i for i in index]
            del intermediate_index[-1]
            intermediate_df = df.iloc[intermediate_index]
            tpm_list.append(intermediate_df.groupby(["SPEED", "acceleration"]).size() / intermediate_df.shape[0])
    tpm_list_final = []
    for df_ in tpm_list:
        try:
            tpm_list_final.append([list(df_.index), [i for i in df_]])
        except:
            tpm_list_final.append(np.nan)
    return tpm_list_final


def tpm_csv(frequency_df, tpm_list_final):
    """
    :param frequency_df: 原始的Frequency_df
    :param tpm_list_final: 对应的每一秒tpm_list及其概率
    :return: 一个完全的df，没有空值，完全的一个状态
    """
    csv_list = []
    for i in tpm_list_final:
        try:
            csv_list.append(tuple(i))
        except:
            csv_list.append(np.nan)
    tpm_df = pd.DataFrame(csv_list, columns=["State", "Probe"])
    return pd.concat([frequency_df,tpm_df],axis=1)


if __name__ == '__main__':
    df,frequency_df = discrete_and_deal_data(read_data(r"D:\Forerunner\data\在建立西宁马尔可夫工况时导入的数据"))
    tpm_list_final = generate_tpm(df,frequency_df)
    tpm_csv(frequency_df,tpm_list_final).to_csv(r".\tpm_and_frequency_df\tpm_.csv")
    frequency_df.to_csv(r".\tpm_and_frequency_df\.frequency_.csv")
