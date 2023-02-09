import cv2 as cv

cap=cv.VideoCapture(0)
face_cascade=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade=cv.CascadeClassifier("haarcascade_eye.xml")
while True:
        _,img=cap.read()
        
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h)in faces:
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv.putText(img,"face",(x,y),cv.FONT_HERSHEY_COMPLEX,1,(250,250,250),1)

        eyes=eye_cascade.detectMultiScale(gray,1.3,5)

        for (x1,y1,w1,h1)in eyes:
            cv.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
            cv.putText(img,"eyes",(x1,y1),cv.FONT_HERSHEY_COMPLEX,0.5,(24,24,23),1)

        cv.imshow('img',img)

        if cv.waitKey(1) == ord('q'):
            break
    
cap.release()
cv.destroyAllWindows()
