import socket
import sys
import threading
import cv2
import time
from os import listdir, mkfifo, remove, unlink, getcwd
from os.path import *
import multiprocessing
import numpy as np
import pickle
from math import isnan
from subprocess import call
from serverReceive import boundary
from constants import *


videoSend = "videoOut.avi"
videoReceive = "videoTemp.avi"
faceDir = "faces"
exitCode = False
notEnoughData = True
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


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

def detect(frame):
    xTemp = 0
    yTemp = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rectsTemp = []
    textTemp = []

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    possibleFaces = []
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        if xTemp < 1 or (((x < xTemp - 25) or (x + w > xTemp + 250)) and ((y < yTemp - 25) or (y + h > yTemp + 250))):
            #rectsTemp.append(((x, y), (x+w, y+h), (255, 255, 255), 1))
            possibleFace = [frame[y:y + h, x:x + w], (x,y),(x+w,y+h)]
            cv2.imwrite('templates/template_' + str(x - x%100) + '_' + str(y - y%100) + '.png', possibleFace[0])
            possibleFaces.append(possibleFace)
    for face, fromCoord, toCoord in possibleFaces:
        faceGrey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        predicted, conf = recognizer.predict(faceGrey)
        print conf
        if conf>threshold:
            rectsTemp.append((fromCoord,toCoord, (0,0,255),1))
            textTemp.append((faceFiles[predicted], fromCoord, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.CV_AA))
        else:
            rectsTemp.append((fromCoord,toCoord, (255,255,255),1))

    while rects:
        rects.pop()#clear the list
    for r in rectsTemp:
        rects.append(r)
    while texts:
        texts.pop()#clear the list
    for t in textTemp:
        texts.append(t)

def setupFiles():
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
def loadData():
    global faceFiles, templates

    faceFiles = [f for f in listdir(faceDir) if isfile(join(faceDir, f))]
    templates = [cv2.imread(join(faceDir, face), 0) for face in faceFiles]

    for i in range(len(faceFiles)-1,-1,-1):
        if templates[i]==None or not templates[i].size:
            templates.pop(i)
            faceFiles.pop(i)

def trainNetwork():
    global recognizer, templates, faceFiles

    recognizer = cv2.createLBPHFaceRecognizer()
    recognizer.train(templates, np.array(range(len(faceFiles))))

def main():
    global rects, texts, templates, faceFiles
    texts, rects = [], []
    faceFiles = [f for f in listdir(faceDir) if isfile(join(faceDir, f))]

    templates = [cv2.imread(join(faceDir, face), 0) for face in faceFiles]
    for i in range(len(faceFiles)-1,-1,-1):
        if templates[i]==None or not templates[i].size:
            templates.pop(i)
            faceFiles.pop(i)
    setupFiles()
    print faceFiles

    global sData, connData, sVideo, connVideo, s, conn, recognizer

    sData, connData = setupSocket(PORT_DATA)
    sVideo, connVideo = setupSocket(PORT_VIDEO)
    s, conn = setupSocket(PORT)


    threading.Thread(target=videoDataReceive).start()
    threading.Thread(target=videoDataSend).start()
    threading.Thread(target=dataReceive).start()

    global cap
    cap = cv2.VideoCapture(videoReceive)
    while not cap.isOpened():
        cap = cv2.VideoCapture(videoReceive)
        time.sleep(.1)
        print("Wait for the header")

    while not isValid(round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT),0)) or not isValid(round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH),0)):
            print("Still waiting for the header")
            time.sleep(.1)

    sourceFPS = 30
    sourceDimensions = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    print("FOURCC:",cap.get(cv2.cv.CV_CAP_PROP_FOURCC))
    loadData()
    trainNetwork()
    while True:
        flag, frame = cap.read()
        if flag:
            detect(frame)
            connData.send(str([list(rects), list(texts)]))
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            time.sleep(.5)
        if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            break

def dataReceive():
    tempBuffer = ""
    while not exitCode:
        data = connData.recv(2**15)
        tempBuffer+=data
        while boundary in tempBuffer:
            ind = tempBuffer.index(boundary)
            rawPart = tempBuffer[:ind]
            tempBuffer = tempBuffer[ind+len(buffer):]
            if not firstPart:
                firstPart = rawPart
            elif not secondPart:
                secondPart = rawPart
            else:
                #Process it
                img = pickle.loads(firstPart)
                fileName = pickle.loads(secondPart)
                cv2.imwrite('faces/' + fileName + '.png', img)
                firstPart,secondPart = None, None
                loadData() #Remove this call and just add it later
                trainNetwork()

    tempFile.close()

def videoDataReceive():
    tempFile = open(videoReceive,"wb")
    while not exitCode:
        data = conn.recv(2**15)
        tempFile.write(data)
        connVideo.send(data)
    tempFile.close()

def videoDataSend():
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
