import cv2
# 导入OpenCV
import numpy as np
# 导入numpy,用于数组的运算和逻辑运算,傅里叶变换和用于图形操作,以及与线性代数相关的操作
import math


#img = cv2.imread('five.JPG')#读入图片
img = cv2.imread('roi_sun.png')#读入图片
img = cv2.resize(img,(0,0),fx=1,fy=1)#按比例缩放图像B
cv2.imshow('img', img)#展示原图
cv2.waitKey(0)#等待输入任意键
cv2.destroyAllWindows()#结束窗口


fgbg = cv2.createBackgroundSubtractorMOG2()  # 利用BackgroundSubtractorMOG2算法消除背景
# fgmask = bgModel.apply(frame)
fgmask = fgbg.apply(img)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
kernel = np.ones((5, 5), np.uint8)
fgmask = cv2.erode(fgmask, kernel, iterations=1)  # 膨胀
res = cv2.bitwise_and(img, img, mask=fgmask)
ycrcb = cv2.cvtColor(res, cv2.COLOR_BGR2YCrCb)  # 分解为YUV图像,得到CR分量
(_, cr, _) = cv2.split(ycrcb)
cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 高斯滤波
_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("skin",skin)
cv2.waitKey(0)#等待输入任意键
cv2.destroyAllWindows()#结束窗口


# 轮廓线绘制
# 现在让我们在图像上找到轮廓。
im, contours, hierarchy = cv2.findContours(skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = max(contours, key=lambda x: cv2.contourArea(x))
cv2.drawContours(img, [contours], -1, (255,255,0), 2)
cv2.namedWindow('contours',cv2.WINDOW_NORMAL)
cv2.imshow("contours", img)
cv2.waitKey(0)#等待输入任意键
cv2.destroyAllWindows()#结束窗口

# 凸包检测
# hull = cv2.convexHull(contours)
# cv2.drawContours(img, [hull], -1, (0, 255, 255), 2)
# cv2.namedWindow('hull',cv2.WINDOW_NORMAL)
# cv2.imshow("hull", img)
# cv2.waitKey(0)
#
# # 凸缺陷检测
# # 手掌与凸包检测轮廓线的任何偏离的地方都可以视为凸度缺陷。
# hull = cv2.convexHull(contours, returnPoints=False)
# defects = cv2.convexityDefects(contours, hull)
# cv2.waitKey(0)

# 通过这一点，我们可以轻松得出Sides：a，b，c（请参见CODE），
# 并且根据余弦定理，我们还可以得出两根手指之间的伽马或角度。
# 如前所述，如果伽玛小于90度，我们会将其视为手指。
# 知道伽玛后，我们只需画一个半径为4的圆，到最远点的近似距离即可。
# 在将文本简单地放入图像中之后，我们就表示手指数（cnt）

# 找出轮廓
im, contours,h = cv2.findContours(skin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours,key=lambda x:cv2.contourArea(x))
epsilon = 0.0005*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
hull = cv2.convexHull(cnt)
areahull = cv2.contourArea(hull)
areacnt = cv2.contourArea(cnt)
arearatio = ((areahull-areacnt)/areacnt)*100
# 求出凹凸点
hull = cv2.convexHull(approx,returnPoints=False)
defects = cv2.convexityDefects(approx,hull)
print(defects.shape[0])
cv2.namedWindow('hull',cv2.WINDOW_NORMAL)
cv2.imshow("hull", img)
cv2.waitKey(0)#等待输入任意键
cv2.destroyAllWindows()#结束窗口

l = 0  # 定义凹凸点个数初始值为0
for i in range(defects.shape[0]):
    s, e, f, d, = defects[i, 0]
    start = tuple(approx[s][0])
    end = tuple(approx[e][0])
    far = tuple(approx[f][0])
    pt = (100, 100)

    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    s = (a + b + c) / 2
    ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
    # 手指间角度求取
    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

    if angle <= 90 and d > 20:
        l += 1
        cv2.circle(skin, far, 3, [255, 0, 0], -1)
    cv2.line(skin, start, end, [0, 255, 0], 2)  # 画出包络线
    print(l)
l += 1
font = cv2.FONT_HERSHEY_SIMPLEX
# 下面的都是条件判断，也就是知道手势后你想实现神么功能就写下面判断里就行了。

if l == 1:
    if areacnt<2000:
        cv2.putText(img,"put hand in the window",(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    else:
        if arearatio<12:
            cv2.putText(img,'0',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        elif arearatio<17.5:
            cv2.putText(img,"1",(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        else:
            cv2.putText(img,'1',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
elif l==2:
    cv2.putText(img,'2',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
elif l==3:
    if arearatio<27:
        cv2.putText(img,'3',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    else:
        cv2.putText(img,'3',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
elif l==4:
    cv2.putText(img,'4',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
elif l==5:
    cv2.putText(img,'5',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
print(l)

cv2.imwrite('C:\\Users\\jklwljcz\\Desktop\\TEST\\FIVE.jpg',img)
cv2.imshow('final_result',img)
cv2.waitKey(0)#等待输入任意键
cv2.destroyAllWindows()#结束窗口


# cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
# cv.namedWindow('final_result',cv.WINDOW_NORMAL)
# cv.imshow('final_result',img)
# cv.waitKey(0)
# cv.destroyAllWindows()