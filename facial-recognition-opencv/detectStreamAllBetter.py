import cv2
import numpy as np
import sys
import tkinter as tk
from os import listdir
from os.path import isfile, join
import glob
import multiprocessing

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


#This simply returns and destroys the text box window
def get_window_text():
    global templateName
    templateName = e.get()
    master.destroy()


#Freezes current process to enter text for clicked image
def create_new_text_window():
    global master
    global e
    master = tk.Tk()
    e = tk.Entry(master)
    e.pack()
    e.focus_set()
    b = tk.Button(master, text="That'll do!", width=10, command=get_window_text)
    b.pack()
    master.mainloop()


#Mouse callback function to get position and click event
def get_mouse_position_onclick(event, ix, iy, flags, param):
    global rects, curFrame, templates, faceFiles
    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, ((x,y),(x2,y2),_,__) in enumerate(rects):
            if (x < ix) and (x2> ix) and (y < iy) and (y2 > iy):
                cv2.imshow('Recognized', curFrame[y:y2,x:x2])
                ix, iy = -1, -1
                create_new_text_window()
                cv2.imwrite('faces/' + templateName + '.png', curFrame[y:y2,x:x2])
                templates= [cv2.imread('faces/' + templateName + '.png', 0)]+templates
                faceFiles =[templateName+".png"]+faceFiles
                break

def detect(frame, faceFiles, templates, sizes, threshold, texts, rects):
    print("In detect")
    xTemp = 0
    yTemp = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rectsTemp = []
    textTemp = []
    # Match a Template
    inner = 0
    for size in sizes:
        for index, template in enumerate(templates):
            tempTemplate = cv2.resize(template, (size, size))
            res = cv2.matchTemplate(gray, tempTemplate, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            for point in zip(*loc[::-1]):
                print("Detected")
                detection = frame[point[1]:point[1] + size, point[0]:point[0] + size]
                # NOTE: its img[y: y + h, x: x + w]
                rectsTemp.append((point, (point[0] + size, point[1] + size), (0, 0, 255), 4))
                personname = faceFiles[index][:-4]
                textTemp.append((personname, (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA))
                xTemp = point[0]
                yTemp = point[1]
                cv2.imwrite('detections/detection_' + str(point[0] - point[0] % 100) + '_' + str(point[1] - point[1] % 100) + '_' + faceFiles[index], detection)
                inner = 1
                break
            if inner == 1:
                break
        if inner == 1:
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
    while rects:
        rects.pop()#clear the list
    for r in rectsTemp:
        rects.append(r)
    while texts:
        texts.pop()#clear the list
    for t in textTemp:
        texts.append(t)
    print(rects)
    print(texts)

def execute_main_loop():
    global ix, iy, ievent, master, templateName, rects, texts, curFrame, templates, faceFiles
    manager = multiprocessing.Manager()
    texts = manager.list()
    rects = manager.list()
    faceDir = "faces"
    faceFiles = [f for f in listdir(faceDir) if isfile(join(faceDir, f))]
    print(faceFiles)
    templates = [cv2.imread(face, 0) for face in faceFiles]
    for i in range(len(faceFiles)-1,-1,-1):
        if templates[i]==None or not templates[i].size:
            templates.pop(i)
            faceFiles.pop(i)

    sizes = [80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
    xTemp = 0
    yTemp = 0
    threshold = 0.4
    video_capture = cv2.VideoCapture(0)
    frameSkipper = 10
    frameIndex = 0
    process = None

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not process or not process.is_alive():
            print("In process")
            if process:
                process.join(1)
            curFrame = frame
            process = multiprocessing.Process(target=detect, args=(frame,faceFiles, templates,sizes, threshold, texts, rects))
            process.start()
            print("Out process")

        for rect in rects:
            cv2.rectangle(frame,*(rect))

        for text in texts:
            cv2.putText(frame,*(text))
        # Display the resulting frame
        cv2.imshow('Video', frame)

        #Quit when q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    #global variables
    template = 0
    sizes = [80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
    threshold = 0.725
    ix, iy = -1, -1
    ievent = 0
    templateName = ''
    master = 0
    e = 0

    #video feed source and windows
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow('Video')
    cv2.namedWindow('Recognized')

    cv2.setMouseCallback('Video', get_mouse_position_onclick)
    execute_main_loop()
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
