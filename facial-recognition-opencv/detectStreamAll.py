import cv2
import sys
import numpy as np
from os import listdir
from os.path import isfile, join
import multiprocessing

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
texts, rect = 0;
def detect(frame, faceFiles, templates, sizes, threshold):
    xTemp = 0
    yTemp = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rectsTemp = []
    textTemp = []
    # Match a Template
    for size in sizes:
        for index, template in enumerate(templates):
            tempTemplate = cv2.resize(template, (size, size))
            res = cv2.matchTemplate(gray, tempTemplate, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            for point in zip(*loc[::-1]):
                detection = frame[point[1]:point[1] + size, point[0]:point[0] + size]
                # NOTE: its img[y: y + h, x: x + w]
                rectsTemp.append((point, (point[0] + size, point[1] + size), (0, 0, 255), 4))
                textTemp.append((faceFiles[index], (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA))
                xTemp = point[0]
                yTemp = point[1]
                cv2.imwrite('detections/detection_' + str(point[0] - point[0]%100) + '_' + str(point[1] - point[1]%100) + '_' + faceFiles[index], detection)
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
        if xTemp < 1 or (((x < xTemp - 25) or (x + w > xTemp + 250)) and ((y < yTemp - 25) or (y + h > yTemp + 250))):
            rectsTemp.append(((x, y), (x+w, y+h), (255, 255, 255), 1))
            possibleFace = frame[y:y + h, x:x + w]
            cv2.imwrite('templates/template_' + str(x - x%100) + '_' + str(y - y%100) + '.png', possibleFace)
    texts.clear()
    for r in rectsTemp:
        rects.append(r)
    rects.clear()
    for t in textTemp:
        texts.append(t)
if __name__ == '__main__':
    global texts, rects
    manager = multiprocessing.Manager()
    texts = manager.list()
    rects = manager.list()
    faceDir = "faces"
    faceFiles = [f for f in listdir(faceDir) if isfile(join(faceDir, f))]
    templates = [cv2.imread(face, 0) for face in faceFiles]
    sizes = [80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
    xTemp = 0
    yTemp = 0
    threshold = 0.7
    video_capture = cv2.VideoCapture(0)
    frameSkipper = 10
    frameIndex = 0
    rects, texts = [],[]
    process = None

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not process or not process.is_alive():
            print("In process")
            if process:
                process.join(1)
            process = multiprocessing.Process(target=detect,
                                          args=(frame,faceFiles, templates,sizes, threshold))
            process.start()

        for rect in rects:
            cv2.rectangle(frame,*(rect))
        for text in texts:
            cv2.putText(frame,*(text))
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()