import cv2
from cvzone.HandTrackingModule import HandDetector

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector=HandDetector(detectionCon=0.8)
startDist=0
scale=0
cx, cy = 500,200

while True:
    success,img=cap.read()
    hands,img=detector.findHands(img)
    img1=cv2.imread("1.png")
    if len(hands)==2:
        #print("1")
        #print(detector.fingersUp(hands[0]),detector.fingersUp(hands[1]))
        if detector.fingersUp(hands[0])==[1,1,0,0,0] and detector.fingersUp(hands[1])==[1,1,0,0,0]:
            #print("1")
            lmList1=hands[0]["lmList"]
            lmList2=hands[1]["lmList"]
            if startDist is None:
                #lmList1[8],lmList2[8]右、左手指尖
 
                # length,info,img=detector.findDistance(lmList1[8],lmList2[8], img)
                length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                startDist=length
            length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
            # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
            scale=int((length-startDist)//2)
            cx,cf=info[4:]       
    else:
        startDist=None        
    try:
         h1, w1, _ = img1.shape
         newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
         img1 = cv2.resize(img1, (newW, newH))
         img[cy-newH//2:cy+newH//2, cx-newW//2:cx+newW//2]=img1
    except:
           pass
    #img[0:250,0:188]=img1
    cv2.imshow("Image",img)
    if cv2.waitKey(1)==ord(' '):
        break