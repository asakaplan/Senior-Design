import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt


cascPath = sys.argv[1]
templatePath = sys.argv[2]
faceCascade = cv2.CascadeClassifier(cascPath)
template = cv2.imread(templatePath, 0)
w, h = template.shape[::-1]

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Match a Template
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.40
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    faces = faceCascade.detectMultiScale(
       gray,
       scaleFactor=1.1,
       minNeighbors=3,
       minSize=(15, 15),
       flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()