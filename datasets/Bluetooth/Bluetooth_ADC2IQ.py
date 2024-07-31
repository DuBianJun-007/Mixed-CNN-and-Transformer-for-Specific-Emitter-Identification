##
# 将源ADC数据转换为IQ数据
##

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# 记录文件读取
file_path = 'Sample_Signal.txt'
data = np.loadtxt(file_path)

# 生成时间序列向量
fs = 250e6  # 采样频率
# fs 必须根据记录文件的采样频率输入 250e6、5e9、10e9 或 20e9。
N = len(data)  # 样本数量
Duration = N / fs  # 信号持续时间
time = np.linspace(0, Duration, N)  # 时间向量
Signal = np.column_stack((data, time))

# 标绘信号
plt.figure()
plt.plot(Signal[:, 1], Signal[:, 0])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Signal Plot')
plt.show()

# I/Q 数据生成
HT = hilbert(Signal[:, 0])
Q = np.imag(HT)  # 积分 (I) 数据
I = np.real(HT)  # 同相 (Q) 数据

# 打印 I 和 Q 数据的前几个值以验证
print("I Data (first 10 samples):", I[:10])
print("Q Data (first 10 samples):", Q[:10])


# 代码说明：
# 导入库：导入 numpy、matplotlib 和 scipy.signal 中的 hilbert 函数。
# 读取记录文件：使用 np.loadtxt 读取文本文件中的数据。
# 生成时间序列向量：计算信号持续时间并生成时间向量，然后将数据和时间向量合并。
# 标绘信号：使用 matplotlib 绘制信号。
# I/Q 数据生成：使用 hilbert 函数生成希尔伯特变换，提取同相 (I) 和正交 (Q) 组件。
# 打印 I 和 Q 数据的前几个值：以验证生成的 I 和 Q 数据。