import cv2
import time
import mediapipe as mp

cap= cv2.VideoCapture(0)

mpHand=mp.solutions.hands
hands=mpHand.Hands() #ellerin tespiti ve konumlandırılması için
mpdraw=mp.solutions.drawing_utils #çizim için

pTime=0
cTime=0

while True:
    control, capture= cap.read()
    results=hands.process(capture)

    print(results.multi_hand_landmarks) #el tespiti koordinatları için

    if results.multi_hand_landmarks: #her bir landmarks içerisinde gezilir
        for handlms in results.multi_hand_landmarks: #handlms ile x, y ve z koordinatları alınır
            mpdraw.draw_landmarks(capture,handlms,mpHand.HAND_CONNECTIONS)

            for number, lm in enumerate(handlms.landmark):
                print(number, lm) #number=eklem numarası, lm=x,y,z koordinatları
                h,w,c=capture.shape
                cx,cy=int(lm.x*w),int(lm.y*h)

                #işaret parmağının tespiti
                # if((number==5)|(number==6)|(number==7)|(number==8)):
                #     cv2.circle(capture,(cx,cy),20,(255,0,0), cv2.FILLED)
                
                #parmak uçlarının tespiti
                if((number==4)|(number==8)|(number==12)|(number==16)|(number==20)):
                    cv2.circle(capture,(cx,cy),20,(255,0,0), cv2.FILLED)

    #fps
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(capture, "FPS: "+ str(int(fps)), (10,75), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),5)
    cv2.imshow("capture",capture)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cv2.destroyAllWindows()