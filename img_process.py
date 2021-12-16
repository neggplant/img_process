
# coding: utf-8

import cv2
import numpy as np
img0 = cv2.imread('bridge.jpg')
img = cv2.imread("fa.jpg")
imgInfo = img0.shape
height = imgInfo[0]
width = imgInfo[1]
img0.shape

# 用于显示图像
while True:
    cv2.imshow('dst', dst)
    if cv2.waitKey(10) == ord("q"):
        break
cv2.destroyAllWindows()

# 压缩方式：https://docs.opencv.org/3.4.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
# 最近邻域和双线性插值
# 默认双线性插值
dst1 = cv2.resize(img, (100, 100), interpolation = cv2.INTER_LINEAR )


#图片剪切直接使用slice
dst = img[100:200, 100:300]
#水平拼接图片
img2 = cv2.hconcat([img0[:, 300:, :], img0[:, :300, :]])
img2 = cv2.vconcat([img0[300:, :, :], img0[:300, :, :]])
cv2.imshow('image', img2)
cv2.waitKey(0)




# 仿射函数cv2.warpAffine()接受三个参数，需要变换的原始图像，移动矩阵M 以及变换的图像大小
matScale = np.float32([[0.5,0,0], [0,0.5,0]])
dst = cv2.warpAffine(img,matScale, (int(width/2), int(height/2)))


# 仿射过程提供原始图片和目标图片(左上角 左下角 右上角)点坐标
matSrc = np.float32([[0, 0], [0, height-1], [width-1, 0]])
matDst = np.float32([[50, 50], [300, height-200], [width-300, 100]])
# 实现放射变换
matAffine = cv2.getAffineTransform(matSrc, matDst)# mat 1 src 2 dst
dst = cv2.warpAffine(img, matAffine, (width, height))

# 图片旋转
# 先设置旋转中心和旋转角度，逆时针旋转,
# 旋转后实际的大小，旋转后会使图像丢失，可将图像缩小，空余部分使用（0,0,0）填充。
matRotate = cv2.getRotationMatrix2D((width*0.5, height*0.5), 10, 0.5)
dst = cv2.warpAffine(img, matRotate, (width, height))


# 灰度处理
dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 使用bgr均值
dst = np.zeros((height, width, 3), np.uint8)
for i in range(0, height):
    for j in range(0, width):
        (b, g, r) = img[i, j]
        gray = (int(b)+int(g)+int(r)) / 3
        dst[i, j] = np.uint8(gray)
# 使用gray = r*0.299+g*0.587+b*0.114
for i in range(0, height):
    for j in range(0, width):
        (b, g, r) = img[i, j]
        b = int(b)
        g = int(g)
        r = int(r)
        gray = r*0.299+g*0.587+b*0.114
        dst[i, j] = np.uint8(gray)


# 马赛克效果，使用某一块的第一个值代替全部
dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for m in range(100, 300):
    for n in range(100, 200):
# pixel ->10*10
if m%10 == 0 and n%10==0:
    for i in range(0, 10):
for j in range(0, 10):
    (b, g, r) = img[m, n]
img[i+m, j+n] = (b, g, r)


# 毛玻璃效果，使用相邻之后的随机像素点代替当前像素
dst = np.zeros((height, width, 3), np.uint8)
mm = 8
for m in range(0, height-mm):
    for n in range(0, width-mm):
        index = int(random.random() * 8)#0-8
        (b,g,r) = img[m+index, n+index]
        dst[m,n] = (b,g,r)

# 图片融合
roiH = int(height/2)
roiW = int(width/2)
img0ROI = img0[roiH:height, roiW:width]
img1ROI = img1[roiH:height, roiW:width]
dst = np.zeros((roiH, roiW, 3), np.uint8)
# 图像占比
dst = cv2.addWeighted(img0ROI, 0.5, img1ROI, 0.5, 0)

# 边缘检测 要使用灰度图。高斯滤波再进行canny边缘检测
# 先进行
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgG = cv2.GaussianBlur(gray, (3,3),0)
dst = cv2.Canny(img, 50, 50)

# sobel 1 算子模版 2 图片卷积 3 阈值判决
# [1 2 1          [ 1 0 -1
#  0 0 0            2 0 -2
# -1 -2 -1 ]       1 0 -1 ]
# [1 2 3 4] [a b c d] a*1+b*2+c*3+d*4 = dst
# sqrt(a*a+b*b) = f>th
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = np.zeros((height, width, 1), np.uint8)
for i in range(0, height-2):
    for j in range(0, width-2):
gy = gray[i,j]*1+gray[i,j+1]*2+gray[i,j+2]*1-gray[i+2,j]*1-gray[i+2,j+1]*2-gray[i+2,j+2]*1
gx = gray[i,j]+gray[i+1,j]*2+gray[i+2,j]-gray[i,j+2]-gray[i+1,j+2]*2-gray[i+2,j+2]
grad = math.sqrt(gx*gx+gy*gy)
# 设置判定阈值，显示边缘图像
if grad>50:
    dst[i,j] = 255
else:
    dst[i,j] = 0

# 浮雕效果 相邻的像素相减后加150，
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# newP = gray0-gray1+150
dst = np.zeros((height, width, 1), np.uint8)
for i in range(0, height):
    for j in range(0, width-1):
        grayP0 = int(gray[i, j])
        grayP1 = int(gray[i, j+1])
        newP = grayP0-grayP1+150
        if newP > 255:
            newP = 255
        if newP < 0:
            newP = 0
        dst[i, j] = newP


# 油画特效
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = np.zeros((height, width, 3), np.uint8)
for i in range(4, height-4):
    for j in range(4, width-4):
        array1 = np.zeros(8, np.uint8)
        for m in range(-4, 4):
            for n in range(-4, 4):
                p1 = int(gray[i+m, j+n]/32)
                array1[p1] = array1[p1]+1
        currentMax = array1[0]
        l = 0
        for k in range(0, 8):
            if currentMax < array1[k]:
                currentMax = array1[k]
                l = k
        # 简化 均值
        for m in range(-4, 4):
            for n in range(-4, 4):
                if gray[i+m,j+n] >= (l*32) and gray[i+m, j+n] <= ((l+1)*32):
                    (b,g,r) = img[i+m, j+n]
        dst[i,j] = (b,g,r)



# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
cv2.line(img,(0,0),(511,511),(255,0,0),5)

cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

cv2.circle(img,(447,63), 63, (0,0,255), -1)

cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

# 图片添加文字
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (10,500), font, 4, (255,255,255), 2, cv2.LINE_AA)

pts = np.array([[10,5], [20,30], [70,20], [50,10]], np.int32)
pts = pts.reshape((-1,1,2))
dst = cv2.polylines(img,[pts], True, (0,255,255))

        # 彩色直方图
def ImageHist(image, type):
    color = (255, 255, 255)
    windowName = 'Gray'
    if type == 31:
        color = (255, 0, 0)
        windowName = 'B Hist'
    elif type == 32:
        color = (0, 255, 0)
        windowName = 'G Hist'
    elif type == 33:
        color = (0, 0, 255)
        windowName = 'R Hist'
    # 1 image 2 [0] 3 mask None 4 256 5 0-255
    hist = cv2.calcHist([image], [0], None, [256], [0.0,255.0])
    minV, maxV, minL, maxL = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256,3], np.uint8)
    for h in range(256):
        intenNormal = int(hist[h]*256/maxV)
        cv2.line(histImg, (h,256), (h,256-intenNormal),color)
    cv2.imshow(windowName, histImg)
    return histImg
img = cv2.imread('image0.jpg', 1)
channels = cv2.split(img)# RGB - R G B
for i in range(0, 3):
    ImageHist(channels[i],31+i)

# cv2.calcHist(images, channels, mask, histSize, ranges[,hist[,accumulate]])

# 1.images:这是uint8或者float32的原图。应该是方括号方式传入：“[img]”
# 2.channels:也是用方括号给出的，我们计算histogram的channel的索引，比如，如果输入时灰度图，值就是[0]，对于彩色图片，
#   你可以传[0],[1]和[2]来分别计算蓝色，绿色和红色通道的histogram。
# 3.mask：掩图，要找到整个图像的histogram，这里传入"None"。但是如果你想找到特定区域图片的histogram，就得创建一个掩图
# 4.histSize：BIN数量，需要用户方括号传入，对于全刻度，我们传入[256].
# 5.ranges：RANGE，一般来说是[0,256].


# 灰度直方图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
count = np.zeros(256, np.float)
for i in range(0, height):
    for j in range(0, width):
        pixel = gray[i, j]
        index = int(pixel)
        count[index] = count[index]+1
for i in range(0, 255):
    count[i] = count[i]/(height*width)
x = np.linspace(0, 255, 256)
y = count
plt.bar(x, y, 0.9, alpha=1, color='b')


# 直方图均衡化 使用255*像素的累计概率
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)

# 彩色直方图均衡化
(b,g,r) = cv2.split(img)#通道分解
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH,gH,rH))# 通道合成


# 图片修复
paint = np.zeros((height,width,1),np.uint8)
for i in range(200, 300):
    paint[i, 200] = 255
    paint[i, 200+1] = 255
    paint[i, 200-1] = 255
for i in range(150, 250):
    paint[250, i] = 255
    paint[250+1, i] = 255
    paint[250-1, i] = 255
cv2.imshow('paint', paint)
#1 src 2 mask
imgDst = cv2.inpaint(img, paint, 3, cv2.INPAINT_TELEA)


import cv2
import numpy as np
img = cv2.imread('bridge.jpg', 1)
cv2.imshow('src', img)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
paint = np.zeros((height, width, 1), np.uint8)


for i in range(200, 300):
    paint[i, 200] = 255
    paint[i, 200+1] = 255
    paint[i, 200-1] = 255
for i in range(150, 250):
    paint[250, i] = 255
    paint[250+1, i] = 255
    paint[250-1, i] = 255
cv2.imshow('paint', paint)
#1 src 2 mask
imgDst = cv2.inpaint(img, paint, 3, cv2.INPAINT_TELEA)

cv2.imshow('image', imgDst)
cv2.waitKey(0)
