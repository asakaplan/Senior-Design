import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt


cascPath = sys.argv[1]
templatePath = sys.argv[2]
faceCascade = cv2.CascadeClassifier(cascPath)
template = cv2.imread(templatePath, 0)
testSizes = [120, 240, 480]
xTemp = 0
yTemp = 0
threshold = 0.3

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    xTemp = 0
    yTemp = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Match a Template
    for size in testSizes:
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for point in zip(*loc[::-1]):
            cv2.rectangle(frame, point, (point[0] + 250, point[1] + 305), (0, 0, 255), 2)
            xTemp = point[0]
            yTemp = point[1]
            #theimage = cv2.getRectSubPix(frame, (point[0], point[1]), (250, 305))

    faces = faceCascade.detectMultiScale(
       gray,
       scaleFactor=1.15,
       minNeighbors=3,
       minSize=(15, 15),
       flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        if xTemp < 1 or (((x < xTemp) or (x > xTemp + 325)) and ((y < yTemp) or (y > yTemp + 325))):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
#theimage = cv2.getRectSubPix(frame,(400, 500),(10, 10))
#cv2.imwrite('person.png', theimage)
video_capture.release()
cv2.destroyAllWindows()