import numpy as np
import pandas as pd
import random
import time
from show_figure import show_figure
import joblib
import random


class ant(object):
    def __init__(self):
        """
        初始化参数
        """
        self.SPEED = 0.0 # 初始速度
        self.acceleration = 0.0 # 初始加速度
        self.path = [] # 选择的路径
        self.max_move_count = 1800 # 最大移动次数
        self.tpm_total = pd.read_csv(r'.\tpm_and_frequency_df\tpm_.csv')
        self.frequency_df = pd.read_csv(r'.\tpm_and_frequency_df\.frequency_.csv').drop("Unnamed: 0",axis=1)
        self.pheromone = 1
        self.path_list = []
        self.loss_list = []

    def choice(self, SPEED, acceleration):
        """
        :param SPEED: 当前速度的状态
        :param acceleration: 当前加速度的状态
        :return: 下一刻的状态
        """
        row = self.tpm_total[(self.tpm_total['SPEED'] == SPEED) & ((self.tpm_total['acceleration'] ==acceleration))]
        row = row.drop("Unnamed: 0",axis=1)
        next_time_state = eval(list(row["State"])[0])
        next_time_probability = eval(list(row["Probe"])[0])
        choice = random.choices(next_time_state,weights=next_time_probability)
        return choice

    def generate_path(self):
        """
        :return:生成的工况
        """
        self.path.append((self.SPEED, self.acceleration))
        for i in range(0,self.max_move_count-1):
            self.SPEED, self.acceleration = self.choice(self.SPEED, self.acceleration)[0]
            self.path.append((self.SPEED, self.acceleration))

    def cal_total_loss(self):
        """
        :return: 计算和原分布的差，损失函数
        """
        df = pd.DataFrame(self.path,columns=["SPEED","acceleration"])  # 生成一个df
        frequency_df = df.groupby(['acceleration', "SPEED"]).size().reset_index(name='Frequency')  # 生成一个频率df
        frequency_df["Frequency"] = frequency_df["Frequency"] / df.shape[0]
        merged_df = pd.merge(frequency_df,self.frequency_df,how="right",left_on=["SPEED","acceleration"],right_on=["SPEED","acceleration"])
        merged_df.fillna(0,inplace=True)
        return np.sum(np.abs(merged_df["Frequency_x"] - merged_df["Frequency_y"])) # 此乃每一条循环的lost

    def adjust(self):
        """
        :return: 只是用来看choice函数是不是正确的
        """
        for i in range(0, 1000):
            v, a = self.choice(0, 0)[0]
            if v != 0 or a != 0:
                print(v)
                print(a)
                break

    def judgement(self):
        """
        :return: 是否保留该次工况。判断条件，怠速时间，终末速度是否为0
        """
        df = pd.DataFrame(self.path,columns=["SPEED","acceleration"])
        if df.iloc[len(df)-1, :][0] <= 0.2: # 条件1，终末速度为0,防止某些似停非停的状况
            if len(df[df["SPEED"] == 0]) < 1000:
                if df["SPEED"].max() < 95:
                    self.path_list.append(df)
                    self.loss_list.append(self.cal_total_loss())
                    print(self.loss_list)
                    df.to_csv(rf"./driving_cycles/{random.randint(1,10**6)}.csv")
        self.acceleration = 0.0
        self.SPEED = 0.0
        self.path = []

    def generate_new_pheromone(self):
        pass

    def run(self):
        """
        :return: 执行该步之后生成50个玩意
        """
        while len(self.path_list) < 2:
            self.generate_path() # 生成类中的path
            self.judgement()


def job_lib_run():
    Ant = ant()
    Ant.run()
    return Ant.path_list


if __name__ == '__main__':
    num_cores = joblib.cpu_count()
    results = joblib.Parallel(n_jobs=-1, backend="loky", prefer="processes")(
        joblib.delayed(job_lib_run)() for i in range(num_cores)
    )
    for result in results:
        print(result)

