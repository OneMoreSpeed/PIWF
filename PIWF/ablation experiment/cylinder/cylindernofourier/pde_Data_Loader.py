"""

time : 2023.7.12
worker ； Sun_Lucheng

"""


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pickle
import torch
import numpy as np

class PDE_input_Dataset(Dataset):
    """
    Class for getting datas and labels
    Args:
        u_v_data_dir = path of input u_v_information(文件夹)
        Interval_step：间隔时间步，默认为1
        #数据处理的图像没必要打乱，与图像识别等工作不同，需要看到所有的信息！！！
        transform_Data = Input Images transformation (default: None)
        transform_Label = Input Labels transformation (default: None)
    Output:
        u_v_data_tetrad = input_data
        u_v_label = u_v label

        #重写整合输入和标签数据，节省数据存储空间

    """

    def __init__(self, u_v_datas_dir, data_type = "pde", Interval_step = 1, transform_Data = None, transform_Label = None):



        self.datas = u_v_datas_dir
        self.pde_name = data_type
        self.all_datas = self.datas[self.pde_name]



    def __len__(self):

        return len(self.all_datas)

    def __getitem__(self, idx):

        data_input = self.all_datas[idx]

        #检查数据读取是否正确
        # print(self.all_tetrad_u_v_datas[idx][0])
        # print(self.all_labels[idx])

        return data_input


class Pde_U_V_Dataset(Dataset):
    """
    Class for getting datas and labels
    Args:
        u_v_data_dir = path of input u_v_information(文件夹)
        Interval_step：间隔时间步，默认为1
        #数据处理的图像没必要打乱，与图像识别等工作不同，需要看到所有的信息！！！
        transform_Data = Input Images transformation (default: None)
        transform_Label = Input Labels transformation (default: None)
    Output:
        u_v_data_tetrad = input_data
        u_v_label = u_v label

        #重写整合输入和标签数据，节省数据存储空间

    """

    def __init__(self, u_v_datas_dir, data_type = "train", Interval_step = 1, transform_Data = None, transform_Label = None):



        self.datas = u_v_datas_dir
        self.input = data_type + "_input"
        self.label = data_type + "_label"

        self.all_datas = self.datas[self.input]
        self.all_labels = self.datas[self.label]



    def __len__(self):

        return len(self.all_datas)

    def __getitem__(self, idx):

        data_input = self.all_datas[idx]

        data_label = self.all_labels[idx]

        #检查数据读取是否正确
        # print(self.all_tetrad_u_v_datas[idx][0])
        # print(self.all_labels[idx])

        return data_input, data_label


if __name__ == '__main__':
    #测试数据加载模块

    file_name = 'ACdataset.pickle'

    # 打开文件，读取数据
    with open(file_name, 'rb') as file:
        data = pickle.load(file)

    batch_size = 2

    # print(data['train_input'].shape)
    # print(data['train_input'])
    # print(data['train_label'].shape)
    # print(data['train_label'])


    test = Pde_U_V_Dataset(data, data_type = "train")
    pde = PDE_input_Dataset(data)
    data_test = DataLoader(test, batch_size = batch_size, shuffle = False)
    data_pde = DataLoader(pde, batch_size=batch_size, shuffle=False)



    for epoch in range(2):

        for batchidx, (data, label) in enumerate(data_test):

            print("data", data.size())
            print(data)
            print("label", label.size())
            print(label)



