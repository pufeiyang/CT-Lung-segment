import os, glob, time
import numpy as np
import nibabel as nib
import cv2
from skimage.filters import frangi

def window_transform(img, win_min, win_max):
    for i in range(img.shape[0]):
        img[i] = 255.0*(img[i] - win_min)/(win_max - win_min)
        min_index = img[i] < 0
        img[i][min_index] = 0
        max_index = img[i] > 255
        img[i][max_index] = 255       
        img[i] = img[i] - img[i].min()
        c = float(255)/img[i].max()
        img[i] = img[i]*c
    return img.astype(np.uint8)

def sigmoid(img, alpha, beta):
    '''
    para img：输入图像。
    para alpha：高亮血管灰度范围。
    para beta: 高亮血管中心灰度。
    '''
    img_max = img.max()
    img_min = img.min()
    return (img_max - img_min) / (1 + np.exp((beta - img) / alpha)) + img_min


def vesseg(image, label):
    # 窗宽调整
    wintrans = window_transform(image, -1350.0, 650.0)
    # nib.Nifti1Image(wintrans, affine).to_filename(save_path+'wintrans.nii.gz')

    # 获取ROI
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    label = cv2.erode(label, kernel)
    roi = wintrans * label
    # nib.Nifti1Image(roi, affine).to_filename(save_path+'roi.nii.gz')

    # 非线性映射
    roi_sigmoid = sigmoid(roi, 20, 95)
    # nib.Nifti1Image(roi_sigmoid, affine).to_filename(save_path+'sigmoid.nii.gz')

    # 第四步：血管增强
    roi_frangi = frangi(roi_sigmoid, sigmas=range(1, 5, 1),
                                alpha=0.5, beta=0.5, gamma=50, 
                                black_ridges=False, mode='constant', cval=0)
    '''
    para image: 输入图像。
    para sigmas: 滤波器尺度，即 np.arange(scale_range[0], scale_range[1], scale_step)。
    para scale_range：使用后的sigma范围。
    para scale_step：sigma的步长。
    para alptha：Frangi校正常数，用于调整过滤器对于板状结构偏差的敏感度。
    para beta：Frangi校正常数，用于调整过滤器对于斑状结构偏差的敏感度。
    para gamma：Frangi校正常数，用于调整过滤器对高方差/纹理/结构区域的敏感度。
    para black_ridges：当为Ture时，过滤去检测黑色脊线；当为False时，检测白色脊线。
    para mode：可选'constant'、'reflect'、'wrap'、'nearest'、'mirror'五种模式，处理图像边界外的值。
    para cval：与mode的'constant'（图像边界之外的值）结合使用。
    '''
    # nib.Nifti1Image(roi_frangi, affine).to_filename(save_path+'frangi.nii.gz')

    # 第五步：自适应阈值分割
    cv2.normalize(roi_frangi, roi_frangi, 0, 1, cv2.NORM_MINMAX)
    thresh = np.percentile(sorted(roi_frangi[roi_frangi > 0]), 95)
    vessel = (roi_frangi - thresh) * (roi_frangi > thresh) / (1 - thresh)
    vessel[vessel > 0] = 1
    vessel[vessel <= 0] = 0
    return vessel

if __name__ == '__main__':
    start = time.time()

    lung = nib.load('DATA3/Series0204_Med.nii.gz') #输入图像
    affine = lung.affine
    lung_img = lung.get_fdata()
    label_img = nib.load('vessel_segment_result/DATA3/Series0204_Med_lung_mask.nii.gz').get_fdata() # 输入肺标签
    vessel = vesseg(lung_img, label_img)
    nib.Nifti1Image(vessel, affine).to_filename('vessel_segment_result/DATA3/Series0204_Med_vessel_mask.nii.gz')

    end = time.time()
    print('process end','time:'+str(end-start))