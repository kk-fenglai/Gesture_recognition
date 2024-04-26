import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
#环境 face
cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector=HandDetector(detectionCon=0.8)
keys = [['Q','W','E','R','T','Y','U','I','O','P'],
        ['A','S','D','F','G','H','J','K','L',';'],
        ['Z','X','C','V','B','N','M',',','.','/']]

class Button():
    def __init__(self,pos,text,size=[50,50]):
        self.pos=pos
        self.size=size
        self.text=text

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
    success,img=cap.read()
    img=detector.findHands(img)
    lmList,bboxInfo=detector.findPosition(img)
    
    img = drawAll(img,buttonList)
    
    
    cv2.imshow("img",img)
    if cv2.waitKey(1)==ord(' '):
        break