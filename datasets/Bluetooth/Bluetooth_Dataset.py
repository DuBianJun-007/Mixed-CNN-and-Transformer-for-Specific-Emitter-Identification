import os
import pickle
import random

import torch
from torch.utils.data import Dataset
import openpyxl


class BluetoothDataset(Dataset):
    def __init__(self, excel_file, sheet_name='Dataset 250 Msps -IQ', mode='train',
                 cache_dir='cache', noise_std=0.1, noise_percentage=0.3, load_in_memory=False):
        self.data = []
        self.targets = []
        self.mode = mode
        self.sheet_name = sheet_name
        self.cache_file = os.path.join(cache_dir, f'{sheet_name}_{mode}_data.cache')
        self.noise_std = noise_std
        self.noise_percentage = noise_percentage
        self.load_in_memory = load_in_memory
        self.memory_data = []

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if os.path.exists(self.cache_file):
            self.load_cache()
        else:
            self.load_data(excel_file)
            self.save_cache()

        if self.load_in_memory:
            self.load_all_data_into_memory()
        # print(self.data[:10])
        # print("-----------------")
        # print(self.targets[:10])

    def load_data(self, excel_file):
        workbook = openpyxl.load_workbook(excel_file)
        worksheet = workbook[self.sheet_name]
        for row in worksheet.iter_rows(min_row=2, values_only=True):
            file_id, file_path, label, label_id = row
            if label.endswith(f"_{self.mode}"):
                self.data.append(file_path)
                self.targets.append(label_id)

    def save_cache(self):
        with open(self.cache_file, 'wb') as cache_file:
            pickle.dump((self.data, self.targets), cache_file)

    def load_cache(self):
        with open(self.cache_file, 'rb') as cache_file:
            self.data, self.targets = pickle.load(cache_file)

    def load_all_data_into_memory(self):
        for file_path in self.data:
            data = torch.load(file_path)
            data = data.unsqueeze(0)
            data = (data - data.min()) / (data.max() - data.min())
            self.memory_data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.load_in_memory:
            data = self.memory_data[idx]
        else:
            file_path = self.data[idx]
            data = torch.load(file_path)
            data = data.unsqueeze(0)
            data = (data - data.min()) / (data.max() - data.min())

        label_id = self.targets[idx]

        # 随机选择数据
        num_elements = data.numel()
        num_noisy_elements = int(num_elements * self.noise_percentage)
        indices = random.sample(range(num_elements), num_noisy_elements)

        # 创建噪声
        noise = torch.zeros_like(data)
        noise_flat = noise.reshape(-1)  # 使用 reshape 而不是 view

        # 随机选择噪声类型
        noise_type = random.choice(['gaussian', 'poisson', 'uniform'])

        if noise_type == 'gaussian':
            noise_flat[indices] = torch.randn(num_noisy_elements) * self.noise_std
        elif noise_type == 'poisson':
            noise_flat[indices] = torch.poisson(torch.ones(num_noisy_elements) * self.noise_std)
        elif noise_type == 'uniform':
            noise_flat[indices] = torch.rand(num_noisy_elements) * 2 * self.noise_std - self.noise_std

        noise_flat = noise_flat.reshape(data.shape)

        # add noise
        noisy_data = data + noise_flat

        return noisy_data, label_id


# 示例用法
if __name__ == "__main__":
    dataset = BluetoothDataset(excel_file='path_to_excel_file.xlsx', mode='train', load_in_memory=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for data, label in data_loader:
        print(data.shape, label)
