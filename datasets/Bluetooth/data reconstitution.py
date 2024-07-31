import os
import torch


def process_txt_files(root_dir, output_dir, chunk_size=50000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sampling_rate_folder in os.listdir(root_dir):
        sampling_rate_path = os.path.join(root_dir, sampling_rate_folder)
        if os.path.isdir(sampling_rate_path):
            for brand_folder in os.listdir(sampling_rate_path):
                brand_path = os.path.join(sampling_rate_path, brand_folder)
                if os.path.isdir(brand_path):
                    for model_folder in os.listdir(brand_path):
                        model_path = os.path.join(brand_path, model_folder)
                        if os.path.isdir(model_path):
                            for phone_folder in os.listdir(model_path):
                                phone_path = os.path.join(model_path, phone_folder)
                                if os.path.isdir(phone_path):
                                    combined_data = []
                                    for txt_file in sorted(os.listdir(phone_path)):
                                        if txt_file.endswith(".txt"):
                                            txt_file_path = os.path.join(phone_path, txt_file)
                                            with open(txt_file_path, 'r') as file:
                                                for line in file:
                                                    i, q = map(float, line.strip().split())
                                                    combined_data.append([i, q])

                                    # 分割数据并保存
                                    output_phone_folder = os.path.join(output_dir, sampling_rate_folder, brand_folder,
                                                                       model_folder, phone_folder)
                                    if not os.path.exists(output_phone_folder):
                                        os.makedirs(output_phone_folder)

                                    chunk_step = 25000  # 设置步进为25000
                                    chunk_count = (len(combined_data) - chunk_size) // chunk_step + 1  # 计算需要生成的文件数量

                                    for i in range(chunk_count):
                                        start_index = i * chunk_step
                                        end_index = start_index + chunk_size
                                        if end_index > len(combined_data):
                                            break  # 如果结束索引超出数据长度，则停止
                                        chunk_data = combined_data[start_index:end_index]
                                        iq_tensor = torch.tensor(chunk_data, dtype=torch.float32).transpose(0,
                                                                                                            1)  # 转换形状为 [2, 50000]
                                        output_tensor_path = os.path.join(output_phone_folder,
                                                                          f"record_{i + 1:03d}_Processed.pt")
                                        torch.save(iq_tensor, output_tensor_path)


# 使用示例
root_dir = "G:\dataset\Bluetooth\ADC2IQ"
output_dir = "G:\dataset\Bluetooth\ADC2IQ_restruct_half_step"
process_txt_files(root_dir, output_dir)
