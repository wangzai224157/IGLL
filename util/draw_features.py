import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np
import torch
import os
def visualize(features, title="img"):
    # 绘制每一层卷积层输出的特征图
    # 每一层的feature_maps的大小为：[N, C, W, H]，分别表示batch、通道数、宽、高
    # 由于这里只有一张图片，所以通过此方法，去掉batch，[N, C, W, H] -> [C, W, H]
    img = torch.squeeze(features)
    # 将tensor转换成numpy类型
    img = img.detach().numpy()
    # 获取特征图的通道数
    channel_num = img.shape[0]
    # 网络中每一层的输出特征，最多绘制12张特征图
    # num = channel_num if channel_num < 8 else 8
    num = channel_num
    fig = plt.figure()
    # 循环绘制
    for i in range(num):
        plt.subplot(8, 8, i + 1)
        # 依次绘制其中的一个通道，img的size为：[C, W, H]
        plt.imshow(img[i, :, :])
        plt.xticks([])
        plt.yticks([])
        # title = "Channel {}".format(i+1)
        # title = i + 1
        # plt.title(title)
        plt.colorbar()
    # 设置每一层特征图的标题
    # plt.title(title)
    plt.show()
def visualizesum(features, title="img"):
    # 绘制每一层卷积层输出的特征图
    # 每一层的feature_maps的大小为：[N, C, W, H]，分别表示batch、通道数、宽、高
    # 由于这里只有一张图片，所以通过此方法，去掉batch，[N, C, W, H] -> [C, W, H]
    img = torch.squeeze(features)
    # 将tensor转换成numpy类型
    img = img.detach().numpy()
    data = np.zeros_like(img)
    num_channels = img.shape[0]
    for i in range(num_channels):
        data[0,:,:] += img[i,:,:]
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    # 依次绘制其中的一个通道，img的size为：[C, W, H]
    plt.imshow(data[0, :, :] / num_channels)
    # 设置每一层特征图的标题
    plt.title(title)

    # 不显示横纵坐标，添加colorbar
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.show()

if __name__=="__main__":
    print("可视化特征")
    path = "/mnt/sda/zhouying/mulu/code/1.18/DMFN-master/test_results/Features_LayerNorm/UN2023-10-18 17:44:35.565288.pt"
    info = torch.load(path, map_location="cpu")
    y = info["y"]
    w = info["w"]
    e = info["e"]

    # visualizesum(y, "Input")
    visualizesum(w, "REATG 4-Residual (LayerNorm)")
    visualizesum(e, "REATG 4-After (LayerNorm)")

    # visualize(y)
    # visualize(w)
    # visualize(e)