import cv2
import sys
import numpy as np

cascPath = sys.argv[1]
templatePath = sys.argv[2]
faceCascade = cv2.CascadeClassifier(cascPath)
template = cv2.imread(templatePath, 0)
sizes = [80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
xTemp = 0
yTemp = 0
threshold = 0.75

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    xTemp = 0
    yTemp = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Match a Template
    for size in sizes:
        tempTemplate = cv2.resize(template, (size, size))
        res = cv2.matchTemplate(gray, tempTemplate, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for point in zip(*loc[::-1]):
            detection = frame[point[1]:point[1] + size, point[0]:point[0] + size]
            # NOTE: its img[y: y + h, x: x + w]
            cv2.rectangle(frame, point, (point[0] + size, point[1] + size), (0, 0, 255), 4)
            cv2.putText(frame, templatePath, (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
            xTemp = point[0]
            yTemp = point[1]
            cv2.imwrite('detections/detection_' + str(point[0] - point[0]%100) + '_' + str(point[1] - point[1]%100) + '_' + templatePath, detection)
            break

    faces = faceCascade.detectMultiScale(
       gray,
       scaleFactor=1.3,
       minNeighbors=3,
       minSize=(30, 30),
       flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        if xTemp < 1 or (((x < xTemp - 50) or (x + w > xTemp + 300)) and ((y < yTemp - 50) or (y + h > yTemp + 300))):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
            possibleFace = frame[y:y + h, x:x + w]
            cv2.imwrite('templates/template_' + str(x - x%100) + '_' + str(y - y%100) + '.png', possibleFace)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()