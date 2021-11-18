import SimpleITK as sitk
from numpy import int16, spacing
from skimage import measure
import time
import numpy as np



def lungmask(vol):
    #获取体数据的尺寸
    size = sitk.Image(vol).GetSize()
    #获取体数据direction
    direction = sitk.Image(vol).GetDirection()
    #获取体数据的空间尺寸
    spacing = sitk.Image(vol).GetSpacing()
    #获得体数据的oringin
    oringin = sitk.Image(vol).GetOrigin()
    #将体数据转为numpy数组
    volarray = sitk.GetArrayFromImage(vol)

    #根据CT值，将数据二值化（一般来说-450以下是空气的CT值）
    num = -450 # 根据CT图像进行微调
    volarray[volarray>=num]=1
    volarray[volarray<=num]=0
    #生成阈值图像
    threshold = sitk.GetImageFromArray(volarray)
    threshold.SetSpacing(spacing)

    #利用种子生成算法，填充空气
    ConnectedThresholdImageFilter = sitk.ConnectedThresholdImageFilter()
    ConnectedThresholdImageFilter.SetLower(0)
    ConnectedThresholdImageFilter.SetUpper(0)
    ConnectedThresholdImageFilter.SetSeedList([(0,0,0),(size[0]-1,size[1]-1,0)])
    
    #得到body的mask，此时body部分是0，所以反转一下
    bodymask = ConnectedThresholdImageFilter.Execute(threshold)
    bodymask = sitk.ShiftScale(bodymask,-1,-1)
    
    #用bodymask减去threshold，得到初步的lung的mask
    temp = sitk.GetImageFromArray(sitk.GetArrayFromImage(bodymask)-sitk.GetArrayFromImage(threshold))
    temp.SetSpacing(spacing)
    
    #利用形态学来去掉一定的肺部的小区域
    bm = sitk.BinaryMorphologicalClosingImageFilter()
    bm.SetKernelType(sitk.sitkBall)
    bm.SetKernelRadius(4) # 微调参数可以消除未分割到的肺部小区域
    bm.SetForegroundValue(1)
    lungmask = bm.Execute(temp)
    
    #利用measure来计算连通域
    lungmaskarray = sitk.GetArrayFromImage(lungmask)
    label = measure.label(lungmaskarray,connectivity=2)
    props = measure.regionprops(label)

    #计算每个连通域的体素的个数
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]
    
    #最大连通域的体素个数，也就是肺部
    #遍历每个连通区域
    index = np.argmax(numPix)
    label[label!=index+1]=0
    label[label==index+1]=1
    
    label = label.astype("int16")
    l = sitk.GetImageFromArray(label)
    l.SetSpacing(spacing)
    l.SetOrigin(oringin)
    l.SetDirection(direction)
    return l

if __name__ == "__main__":

    start = time.time()
    
    lung = sitk.ReadImage("DATA3/Series0204_Med.nii.gz") # 输入图像
    spacing = lung.GetSpacing()
    direction = lung.GetDirection()
    oringin = lung.GetOrigin()
    print('spacing:',spacing)
    print('direction:',direction)
    print('oringin:',oringin)
    
    array = sitk.GetArrayFromImage(lung).astype(int16)
    new_lung = sitk.GetImageFromArray(array)
    
    new_lung.SetSpacing(spacing)
    new_lung.SetDirection(direction)
    new_lung.SetOrigin(oringin)
    
    lung_mask = lungmask(new_lung)
    print(lung_mask.GetOrigin())
    sitk.WriteImage(lung_mask,"vessel_segment_result/DATA3/Series0204_Med_lung_mask.nii.gz") # 保存图像

    end = time.time()
    print('process end','time:'+str(end-start))