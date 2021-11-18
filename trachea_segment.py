import numpy as np
import nibabel as nib
import time,os
import cv2
from scipy.ndimage import binary_fill_holes

def regionGrowing(grayImg,seed,threshold):
    """
    :param grayImg: 灰度图像
    :param seed: 生长起始点的位置
    :param threshold: 阈值
    :return: 取值为{0, 255}的二值图像
    """
    [maxX, maxY,maxZ] = grayImg.shape[0:3]

    # 用于保存生长点的队列
    pointQueue = []
    pointQueue.append((seed[0], seed[1],seed[2]))
    outImg = np.zeros_like(grayImg)
    outImg[seed[0], seed[1],seed[2]] = 1

    pointsNum = 1
    pointsMean = float(grayImg[seed[0], seed[1],seed[2]])

    # 用于计算生长点周围26个点的位置
    Next26 = [[-1, -1, -1],[-1, 0, -1],[-1, 1, -1],
                [-1, 1, 0], [-1, -1, 0], [-1, -1, 1],
                [-1, 0, 1], [-1, 0, 0],[-1, 0, -1],
                [0, -1, -1], [0, 0, -1], [0, 1, -1],
                [0, 1, 0],[-1, 0, -1],
                [0, -1, 0],[0, -1, 1],[-1, 0, -1],
                [0, 0, 1],[1, 1, 1],[1, 1, -1],
                [1, 1, 0],[1, 0, 1],[1, 0, -1],
                [1, -1, 0],[1, 0, 0],[1, -1, -1]]

    while(len(pointQueue)>0):
        # 取出队首并删除
        growSeed = pointQueue[0]
        del pointQueue[0]

        for differ in Next26:
            growPointx = growSeed[0] + differ[0]
            growPointy = growSeed[1] + differ[1]
            growPointz = growSeed[2] + differ[2]

            # 是否是边缘点
            if((growPointx < 0) or (growPointx > maxX - 1) or
               (growPointy < 0) or (growPointy > maxY - 1) or (growPointz < 0) or (growPointz > maxZ - 1)) :
                continue

            # 是否已经被生长
            if(outImg[growPointx,growPointy,growPointz] == 1):
                continue

            data = grayImg[growPointx,growPointy,growPointz]
            # 判断条件
            # 符合条件则生长，并且加入到生长点队列中
            if(abs(data - pointsMean)<threshold):
                pointsNum += 1
                pointsMean = (pointsMean * (pointsNum - 1) + data) / pointsNum
                outImg[growPointx, growPointy,growPointz] = 1
                pointQueue.append([growPointx, growPointy,growPointz])

    return outImg

# def FillHole(mask):
#     mask = np.array(mask, np.uint8)
#     contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
#     # len_contour = len(contours)
#     # print(len_contour)
#     # contour_list = []
#     # for i in range(len_contour):
#     #     drawing = np.zeros_like(mask, np.uint8)  # create a black image
#     #     img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
#     #     contour_list.append(img_contour)
#     # out = sum(contour_list)
#     drawing = mask.copy()  # create a black image
#     drawing = cv2.cvtColor(drawing, cv2.COLOR_GRAY2RGB)
#     out = cv2.drawContours(drawing, contours, -1, (255, 255, 255), -1)
#     out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)

#     return out


if __name__ == '__main__':
    
    start = time.time()
    
    lung = nib.load('DATA3/Series0204_Med.nii.gz') # 输入图像
    lung_img = lung.get_fdata()
    affine = lung.affine
    
    seed = (261,273,324) # 选择合适的种子点，在这里选择气管的位置
    threshold = 80 # 临近像素的阈值，微调可以减少误分割，和少分割
    trachea_img = regionGrowing(lung_img, seed, threshold)
    nib.Nifti1Image(trachea_img, affine).to_filename('vessel_segment_result/DATA3/Series0204_Med_trachea_mask.nii.gz') # 保存图像
    
    end = time.time()
    print('process end','time:',str(end-start))