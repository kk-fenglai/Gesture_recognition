import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import cvzone
from pynput.keyboard import Key,Controller
#环境 face opencvpython=4.5
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
#识别手势
detector = HandDetector(detectionCon=0.8)
keyboard = Controller()
#键盘关键字
keys = [['Q','W','E','R','T','Y','U','I','O','P'],
        ['A','S','D','F','G','H','J','K','L',';'],
        ['Z','X','C','V','B','N','M',',','.','/']]
class Button():
    def __init__(self,pos,text,size = [50,50]):
        self.pos = pos
        self.text = text
        self.size = size
 
buttonList = []
finalText = ''
for j in range(len(keys)):
    for x,key in enumerate(keys[j]):
        #循环创建buttonList对象列表
        buttonList.append(Button([60*x+20,100+j*60],key))
 
def drawAll(img,buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img,(x,y,w,h),20,rt = 0)
        cv2.rectangle(img, (x,y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 10, y + 40),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    return img
 
while True:
    success,img = cap.read()
    #识别手势
    img = detector.findHands(img)
    lmList,bboxInfo = detector.findPosition(img)
 
    img = drawAll(img,buttonList)
    if lmList:
        for button in buttonList:
            x,y = button.pos
            w,h = button.size
            if x<lmList[8][0]<x+w and y<lmList[8][1]<y+h:
                cv2.rectangle(img, (x-5,y-5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 10, y + 40),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
 
                l,_,_ = detector.findDistance(8,12,img,draw=False)
                print('中指(12)和食指(8)之间的距离：',l)
                if l < 50:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 10, y + 40),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
                    finalText += button.text
                    print('当前选中的是：', button.text)
                    sleep(0.2)
    cv2.rectangle(img, (20,350), (600, 400), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (20, 390),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)
    cv2.imshow("Image",img)
    if cv2.waitKey(1)==ord(' '):
        break