import cv2
cam = cv2.VideoCapture(0)
cam.set(3, 660)
cam.set(4, 500)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')


while True :
    retV, frame = cam.read()
    warna = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale (warna, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 63),2)
        rec_muka = warna [y:y+h, x:x+w]
    cv2.imshow('Webcam',frame)
    exit = cv2.waitKey(1) & 0xff
    if exit == 27 or exit == ord('q'):
        break
        cam.release()
        cv2.destroyAllWindows()