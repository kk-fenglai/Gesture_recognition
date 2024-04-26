import cv2 as cv
import mediapipe as mp
import numpy as np
from time import sleep
from pynput.keyboard import Controller, Key
 #环境 test
'''
# 后续会控制鼠标
from pynput.mouse import Controller as mouseController
from pynput.mouse import Button
'''
 
# 设置显示分辨率
class displaySolutionSet():
    def __init__(self, capture, width, height):
        self.capture = capture
        self.width = width
        self.height = height
 
    def solution_set(self):
        self.capture.set(3, self.width)
        self.capture.set(4, self.height)
 
#定义显示键盘的内容
keyText = [["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "+"],
           ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]"],
           ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "ENTER"],
           ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "BACK"]]
 
keyboard = Controller()
 
# 设置半透明效果,这部分写的麻烦了，估计用addWeighted会更好
def translucent(image, post1, post2, BChannel=0, GChannel=255, RChannel=0, value=0.8):
    channel = [BChannel, GChannel, RChannel]
    for i in range(3):
        image[post1[1]: post2[1], post1[0]:post2[0], i] = \
             image[post1[1]: post2[1], post1[0]:post2[0], i]*value + channel[i]*(1-value)
    return image
 
# 绘制键盘
def draw_keyboard(image, rectPos1 = (50, 50), rectPos2 = (120, 120), textPos = (62, 107), textColor=(255, 255, 255)):
    for i, list in enumerate(keyText):
        for j, l in enumerate(list):
            if i == 2 and j == 10:
                image = translucent(image, (rectPos1[0]+90*j, rectPos1[1]+90*i), (rectPos2[0]+90*(j+1), rectPos2[1]+90*i), value=0.7)
                cv.putText(image, l, (textPos[0]+90*j, textPos[1]+90*i), 
                           fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=2, color=textColor, thickness=2)
            elif i== 3 and j == 10:
                image = translucent(image, (rectPos1[0]+90*j, rectPos1[1]+90*i), (rectPos2[0]+90*(j+1), rectPos2[1]+90*i), value=0.7)
                cv.putText(image, l, (textPos[0]+90*j, textPos[1]+90*i), 
                           fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=2, color=textColor, thickness=2)
            else:
                image = translucent(image, (rectPos1[0]+90*j, rectPos1[1]+90*i), (rectPos2[0]+90*j, rectPos2[1]+90*i), value=0.7)
                cv.putText(image, l, (textPos[0]+90*j, textPos[1]+90*i), 
                           fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3, color=textColor, thickness=2)
#  检测是否按下按键
def click_detect(point1, point2, point3):
    result = 0
    #计算向量的L2范数
    dist1 = np.linalg.norm((point2 - point1), ord=2)
    dist2 = np.linalg.norm((point3 - point1), ord=2)
    if dist2 > dist1:
        result = 1
        print(dist2)
    else:
        result = 0
    return result
 
#返回按键的值并输出
def key_value(image, point, clicked):
    xPos, yPos = point[0], point[1]
    for i, list in enumerate(keyText):
        for j, l in enumerate(list):
            # 判断手指是否移动到某个按键的区域内
            if ((50+90*j) < xPos < (120+90*j)) and ((50+90*i) < yPos < (120+90*i)):
                # 回车键单独处理
                if i == 2 and j == 10:
                    translucent(image, post1=(50+90*j, 50+90*i), post2=(120+90*(j+1), 120+90*i), value=0)
                    cv.putText(image, l, (60+90*j, 107+90*i), 
                               fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 255, 255), thickness=3)
                    if clicked:
                        keyboard.press(Key.enter)
                        cv.putText(image, l, (60+90*j, 107+90*i), 
                                   fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 0), thickness=3)
                        keyboard.release(Key.enter)
                        sleep(0.1)
                # 退格键单独处理
                elif i == 3 and j == 10 and click_detect:
                    translucent(image, post1=(50+90*j, 50+90*i), post2=(120+90*(j+1), 120+90*i), value=0)
                    cv.putText(image, l, (60+90*j, 107+90*i), 
                               fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 255, 255), thickness=3)
                    if clicked:
                        keyboard.press(Key.backspace)
                        cv.putText(image, l, (60+90*j, 107+90*i),
                               fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 0), thickness=3)
                        keyboard.release(Key.space)
                        sleep(0.1)                    
                else:
                    translucent(image, post1=(50+90*j, 50+90*i), post2=(120+90*j, 120+90*i), value=0)
                    cv.putText(image, l, (60+90*j, 107+90*i), 
                            fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=4, color=(255, 255, 255), thickness=3)
                    if clicked:
                        keyboard.press(l)
                        cv.putText(image, l, (60+90*j, 107+90*i), 
                                   fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=4, color=(255, 0, 0), thickness=3)
                        keyboard.release(l)
                        sleep(0.1)
def main():
    cap = cv.VideoCapture(0)
    landmark = np.arange(42).reshape(21, -1)
 
    if not cap.isOpened():
        print("can not open video capture, please check the number of camera device.\n")
        exit()
 
    solution = displaySolutionSet(cap, width=1280, height=720)
    solution.solution_set()
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=int(10))
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=int(5))   
 
    while True:
        ret, img = cap.read()
        if not ret:
            print("Can not receive frame (stream end?). Exiting...")
            break
        
        # 翻转图像
        img = cv.flip(img, 1)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)                 
        result = hands.process(imgRGB)
        draw_keyboard(img)
        if result.multi_hand_landmarks:
            for i,handLms in enumerate(result.multi_hand_landmarks):
                mpDraw.draw_landmarks(img,
                                      handLms,
                                      mpHands.HAND_CONNECTIONS,
                                      landmark_drawing_spec = handLmsStyle,
                                      connection_drawing_spec = handConStyle)
                
                for j, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * solution.width)
                    yPos = int(lm.y * solution.height)
                    landmark_ = [xPos, yPos]
                    landmark[j, :] = landmark_
 
                click = click_detect(landmark[11], landmark[4], landmark[3])
                key_value(img, landmark[8], click)  
 
        cv.imshow("img", img)
        if cv.waitKey(1) == ord(' '):
            break
 
    cap.release()
    cv.destroyAllWindows()
 
if __name__ == '__main__':
    main()