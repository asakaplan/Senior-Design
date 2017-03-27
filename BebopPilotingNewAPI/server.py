import socket
import sys
import threading
import cv2
import time
from os import listdir, mkfifo, remove, unlink, getcwd
from os.path import *
import multiprocessing
import numpy as np
from math import isnan
from subprocess import call

HOST = ''
PORT = 8080
PORT_VIDEO = 8001
PORT_DATA = 8002
threshold = 0.725

videoSend = "videoOut.avi"
videoReceive = "videoTemp.avi"
exitCode = False
notEnoughData = True
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
sizes = [80, 100, 120, 140, 160, 180, 200, 220, 240, 260]

def setupSocket(port):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    print( 'Socket created')
    try:
            soc.bind((HOST,port))
    except socket.error as msg:
            print('Bind failed. Error Code : ' + str(msg))
            sys.exit()
    print('Socket bind complete')
    soc.listen(10)
    print('Socket now listening on port %d'%port)
    conn, addr = soc.accept()
    print('Connected with ' + addr[0] + ":" + str(addr[1]))
    return soc, conn

def isValid(val):
    return bool(val) and not isnan(val)

def detect(frame, faceFiles, templates, sizes, threshold, texts, rects):
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
                print("Detected:", len(point))
                detection = frame[point[1]:point[1] + size, point[0]:point[0] + size]
                # NOTE: its img[y: y + h, x: x + w]
                rectsTemp.append((point, (point[0] + size, point[1] + size), (0, 0, 255), 4))
                personname = faceFiles[index][:-4]
                textTemp.append((personname, (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.CV_AA))
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

def main():
    global rects, texts, templates, faceFiles
    manager = multiprocessing.Manager()
    texts = manager.list()
    rects = manager.list()
    faceDir = "faces"
    faceFiles = [f for f in listdir(faceDir) if isfile(join(faceDir, f))]
    print(faceFiles)
    templates = [cv2.imread(join(faceDir, face), 0) for face in faceFiles]
    for i in range(len(faceFiles)-1,-1,-1):
        if templates[i]==None or not templates[i].size:
            templates.pop(i)
            faceFiles.pop(i)
    try:
       remove(videoSend)
    except Exception:
       pass
    try:
        remove(videoReceive)
    except Exception:
        pass
    mkfifo(videoSend)
    mkfifo(videoReceive)
    cwd = getcwd()
    print "perl -MFcntl -e 'fcntl(STDIN, 1031, 524288) or die $!' <> %s"%join(cwd,videoSend)

    call(["perl -MFcntl -e 'fcntl(STDIN, 1031, 524288) or die $!' <> %s"%join(cwd,videoSend)], shell=True)
    call(["perl -MFcntl -e 'fcntl(STDIN, 1031, 524288) or die $!' <> %s"%join(cwd,videoReceive)], shell=True)
    global sData, connData
    sData, connData = setupSocket(PORT_DATA)
    global sVideo, connVideo
    sVideo, connVideo = setupSocket(PORT_VIDEO)
    global s, conn
    s, conn = setupSocket(PORT)

    i = 0

    threading.Thread(target=dataReceive).start()
    threading.Thread(target=dataSend).start()

    global cap
    cap = cv2.VideoCapture(videoReceive)
    while not cap.isOpened():
        cap = cv2.VideoCapture(videoReceive)
        time.sleep(.1)
        print("Wait for the header")

    while not isValid(round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT),0)) or not isValid(round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH),0)):
            print (round(cap.get(cv2.cv.CV_CAP_PROP_FPS),0), round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT),0), round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH),0))
            print("Still waiting for the header")
            time.sleep(.1)
    print(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    sourceFPS = 30#int(round(cap.get(cv2.cv.CV_CAP_PROP_FPS),0))
    sourceDimensions = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    print("FOURCC:",cap.get(cv2.cv.CV_CAP_PROP_FOURCC))
    global outputVideo
    #outputVideo = cv2.VideoWriter(videoSend, cv2.cv.CV_FOURCC(*'XVID'), sourceFPS, sourceDimensions, 1)
    print("Success: ", outputVideo.isOpened())
    process = None
    pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    i=0
    skipFrames = 10
    while True:
        flag, frame = cap.read()
        if flag:
            if not process or not process.is_alive():
                if process:
                    process.join(1)
                curFrame = frame
                process = multiprocessing.Process(target=detect, args=(frame, faceFiles, templates, sizes, threshold, texts, rects))
                process.start()
                connData.send(str([list(rects), list(texts)]))
            # The frame is ready and already captured
            #cv2.imshow('video', frame)
            pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

            #if not i%skipFrames:
            #    outputVideo.write(frame)
            i+=1
            pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            if(pos_frame%1000==0):print(str(pos_frame)+" frames")
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            time.sleep(.5)
        if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break



def dataReceive():
    tempFile = open(videoReceive,"wb")
    tempFile.write(tempHeader)
    while not exitCode:
        data = conn.recv(2**15)
        tempFile.write(data)
        connVideo.send(data)
    tempFile.close()

def dataSend():
    lastLength = 0
    outp = None
    while not exitCode:
        print("Trying to open output file")
        try:
            outp = open(videoSend, "r+b")
            break
        except Exception as e:
            time.sleep(.1)

    for line in outp:
        #print("Sending data")
        #outp is a fifo, so this will continue to go until the program is exited
        if exitCode:break
        connVideo.send(line)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("In Interrupt")
        exitCode = True
        unlink(videoSend)
        s.close()
        cap.release()
        outputVideo.release()
        cv2.destroyAllWindows()
        print("End of interrupt")
        raise
