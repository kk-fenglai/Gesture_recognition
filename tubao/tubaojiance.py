import cv2
# 导入OpenCV
import numpy as np
# 导入numpy,用于数组的运算和逻辑运算,傅里叶变换和用于图形操作,以及与线性代数相关的操作
import math
#环境classic
cap=cv2.VideoCapture(0)
cnt=0
flag=0
def binaryMask(frame, x0, y0, width, height):
	cv2.rectangle(frame,(x0,y0),(x0+width, y0+height),(0,255,0)) #画出截取的手势框图
	roi = frame[y0:y0+height, x0:x0+width] #获取手势框图
	#roi = topolar(roi)
	#cv2.imshow("roi", roi) #显示手势框图
	res,ret = skinMask(roi) #进行肤色检测
	#cv2.imshow("res", res) #显示肤色检测后的图像
	'''
	kernel = np.ones((3,3), np.uint8) #设置卷积核
	res = cv2.erode(res, kernel) #腐蚀操作
	res = cv2.dilate(res, kernel)#膨胀操作
	cv2.imshow("res", res) #显示肤色检测后的图像
	'''
	#ret = fd.reconstruct(res, fourier_result, MIN_DESCRIPTOR)
	return roi,res,ret


####YCrCb颜色空间的Cr分量+Otsu法阈值分割算法
def skinMask(roi):
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	cr1 = cv2.GaussianBlur(cr, (5,5), 0)
	flag, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu处理
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res,flag

def lunkuo(img):
    img = cv2.resize(img,(0,0),fx=1,fy=1)
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
    return skin

def verify(cnt,img):
    epsilon = 0.0005*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    hull = cv2.convexHull(cnt)
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)
    arearatio = ((areahull-areacnt)/areacnt)*100
    hull = cv2.convexHull(approx,returnPoints=False)
    defects = cv2.convexityDefects(approx,hull)
    
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
    l += 1
    font = cv2.FONT_HERSHEY_SIMPLEX 
# 下面的都是条件判断，也就是知道手势后你想实现神么功能就写下面判断里就行了。
    if l== 1:
     if areacnt<2000:
        cv2.putText(img,"put hand in the window",(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
     else:
        if arearatio<12:
            cv2.putText(img,'0',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        elif arearatio<17.5:
            cv2.putText(img,"1",(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        else:
            cv2.putText(img,'1',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    if l==2:
        cv2.putText(img,'2',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    if l==3:
     if arearatio<27:
        cv2.putText(img,'3',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
     else:
        cv2.putText(img,'3',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    if l==4:
        cv2.putText(img,'4',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    if l==5:
        cv2.putText(img,'5',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    
while True:
    ret,img1=cap.read()
    img = cv2.flip(img1, 2)
    roi,skin,flag=binaryMask(img, 300, 100, 400, 400)
    cv2.imshow("skin",skin)
    #绘制轮廓
    roi_sun= cv2.imread('roi_sun.png')
    lunkou_img=lunkuo(roi_sun)
    cv2.imshow("lunkuo",lunkou_img)
    #线绘制
    im, contours, hierarchy = cv2.findContours(lunkou_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(roi_sun, [contours], -1, (255,255,0), 2)
    cv2.imshow("roi_sun",roi_sun)
       
    """ if(flag==0):
       verify(contours,img)
       flag=1 """
    #寻找凹凸点
    epsilon = 0.0005*cv2.arcLength(contours,True)
    approx = cv2.approxPolyDP(contours,epsilon,True)
    hull = cv2.convexHull(contours)
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(contours)
    arearatio = ((areahull-areacnt)/areacnt)*100
    hull = cv2.convexHull(approx,returnPoints=False)
    defects = cv2.convexityDefects(approx,hull)
    #计算手指个数
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
    l += 1
    font = cv2.FONT_HERSHEY_SIMPLEX 
# 下面的都是条件判断，也就是知道手势后你想实现神么功能就写下面判断里就行了。
    if l== 1:
     if areacnt<2000:
        cv2.putText(img,"put hand in the window",(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
     else:
        if arearatio<12:
            cv2.putText(img,'0',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
            cv2.putText(lunkou_img,'0',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        elif arearatio<17.5:
            cv2.putText(img,"1",(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
            cv2.putText(lunkou_img,'1',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        else:
            cv2.putText(img,'1',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
            cv2.putText(lunkou_img,'1',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    if l==2:
        cv2.putText(img,'2',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        cv2.putText(lunkou_img,'2',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    if l==3:
     if arearatio<27:
        cv2.putText(img,'3',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        cv2.putText(lunkou_img,'3',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
     else:
        cv2.putText(img,'3',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        cv2.putText(lunkou_img,'3',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    if l==4:
        cv2.putText(img,'4',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        cv2.putText(lunkou_img,'4',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    if l==5:
        cv2.putText(img,'5',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        cv2.putText(lunkou_img,'5',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    if(l!=1 and l!=2 and l!=3 and l!=4 and l!=5 ):
        cv2.putText(img,'none',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
        cv2.putText(lunkou_img,'none',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    key = cv2.waitKey(1)#按键判断并进行一定的调整
            #按'j''l''u''j'分别将选框左移，右移，上移，下移
		#按'q'键退出录像
    if key == ord('s') and flag>120:
        name = str(cnt)
        cv2.imwrite('roi_sun.png',roi)
        cnt += 1
    if key==ord(' '):
        break
    print(flag)
    cv2.imshow('img',img)
    #cv2.imshow('img',skin)
cap.release()
cv2.destroyAllWindows()