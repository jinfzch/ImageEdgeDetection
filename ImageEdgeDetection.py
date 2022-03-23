import numpy as np
from PIL import Image
import math

# 定义灰度差矩阵
def blurring(arr):
    for i in range(3):
        for j in range(3):
            if arr[i][j] <= 0.6 and arr[i][j] >=  -0.6:
                arr[i][j] = math.exp(-20 * arr[i][j]*arr[i][j])
            else:
                arr[i][j] = 0
    w1 = min(arr[0,1], arr[1][2])
    w2 = min(arr[1,2], arr[2][1])
    w3 = min(arr[2,1], arr[1][0])
    w4 = min(arr[1,0], arr[0][1])
    B = min(min(1 - w1, 1 - w2), min(1 - w3, 1 - w4))
    return (w1,w2,w3,w4,B)

# 图片导入与处理
img1: Image.Image = Image.open("./图片.png")

img1 = img1.convert("RGB")
np_array = np.array(img1)

col = img1.size[0]
row = img1.size[1]
print(row,col)

# 灰度定义
GrayArray = []
GrayArrayMax = []
GrayArrayRed = []
GrayArrayGreen = []
GrayArrayBlue = []

for itemRow in np_array:
    for itemCol in itemRow:
        GrayArray.append(int(((itemCol[0])*0.299+(itemCol[1])*0.587+(itemCol[2])*0.114)/3))
        GrayArrayMax.append(max(itemCol[0],itemCol[1],itemCol[2]))
        GrayArrayRed.append(itemCol[0])
        GrayArrayGreen.append(itemCol[1])
        GrayArrayBlue.append(itemCol[2])


# 选取灰度算法
np_GrayArrayMax = np.array(GrayArray).reshape(row,col)
img = Image.fromarray(np_GrayArrayMax).convert("L")
img.show()

# 数组中最大最小值
Min = np.min(np_GrayArrayMax)
Max = np.max(np_GrayArrayMax)


# 创建归一化数组
np_EdgeArrayMax = np.zeros((row+2,col+2),dtype=float)
# 数组填充
for i in range(row):
    for j in range(col):
        np_EdgeArrayMax[i+1][j+1] = np_GrayArrayMax[i][j]

# 展示填充效果
img = Image.fromarray(np_EdgeArrayMax).convert("L")
img.show()

# 数组归一化
for i in range(row+2):
    for j in range(col+2):
        np_EdgeArrayMax[i][j] = (np_EdgeArrayMax[i][j] - 0) / (Max-0)

# 创建小矩阵
ProcessArrayMax = np.zeros((3,3),dtype=float)
# 创建处理后的矩阵
DealArrayMax = np.zeros((row+2,col+2),dtype=float)

for i in range(2,row+1):
    for j in range(2,col + 1):
        ProcessArrayMax[0][0] = np_EdgeArrayMax[i-1][j-1] - np_EdgeArrayMax[i][j]
        ProcessArrayMax[0][1] = np_EdgeArrayMax[i-1][j] - np_EdgeArrayMax[i][j]
        ProcessArrayMax[0][2] = np_EdgeArrayMax[i-1][j+1] - np_EdgeArrayMax[i][j]
        ProcessArrayMax[1][0] = np_EdgeArrayMax[i][j-1] - np_EdgeArrayMax[i][j]
        ProcessArrayMax[1][1] = np_EdgeArrayMax[i][j] - np_EdgeArrayMax[i][j]
        ProcessArrayMax[1][2] = np_EdgeArrayMax[i][j+1] - np_EdgeArrayMax[i][j]
        ProcessArrayMax[2][0] = np_EdgeArrayMax[i+1][j-1] - np_EdgeArrayMax[i][j]
        ProcessArrayMax[2][1] = np_EdgeArrayMax[i+1][j] - np_EdgeArrayMax[i][j]
        ProcessArrayMax[2][2] = np_EdgeArrayMax[i+1][j+1] - np_EdgeArrayMax[i][j]
        W1,W2,W3,W4,B = blurring(ProcessArrayMax)
        V1 = 0.8 * W1 + 0.2
        V2 = 0.8 * W2 + 0.2
        V3 = 0.8 * W3 + 0.2
        V4 = 0.8 * W4 + 0.2
        V5 = 0.8 - (0.8 * B)
        DealArrayMax[i][j] = ((W1 * V1) + (W2 * V2) + (W3 * V3) + (W4 * V4) + (B * V5)) / (W1 + W2 + W3 + W4 + B)

print(DealArrayMax)

for i in range(row+2):
    for j in range(col+2):
        DealArrayMax[i][j] = DealArrayMax[i][j] * 255
img = Image.fromarray(DealArrayMax).convert("L")
img.show()
