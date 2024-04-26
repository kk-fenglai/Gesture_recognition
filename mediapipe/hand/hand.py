import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands#手势
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
handLmsStyle=mpDraw.DrawingSpec(color=(0,0,255),thickness=5)
handConStyle=mpDraw.DrawingSpec(color=(0,255,0),thickness=5)
pTime=0
cTime=0


while True:
     ret,img=cap.read()
     if ret:
         imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
         result=hands.process(imgRGB)
         #print(result.multi_hand_landmarks)
         imgHeight=img.shape[0]
         imgWidth=img.shape[1]
         if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:#画出手
             mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS,handLmsStyle,handConStyle)
             for i,lm in enumerate(handLms.landmark):
                 xPos=int(lm.x*imgHeight)
                 yPos=int(lm.y*imgWidth)
                 #cv2.putText(img,str(i),(xPos,yPos),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.4,(0,0,255),2)
                 print(i,xPos,yPos)
     cTime=time.time()
     fps = 1/(cTime-pTime)
     pTime=cTime
     cv2.putText(img,f"fps:{int(fps)}",(30,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
     cv2.imshow('img',img)
     if cv2.waitKey(1)==ord(' '):
        break
cap.release()
cv2.destroyAllWindows()