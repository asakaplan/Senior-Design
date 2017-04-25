import socket
import sys
import threading
import cv2
import time
from os import listdir, mkfifo, remove, unlink, getcwd, mkdir
from os.path import *
import multiprocessing
import numpy as np
import pickle
from math import isnan
from subprocess import call
from serverReceive import boundary
from constants import *
import Tkinter as tk
from os import listdir
from os.path import isfile, join, dirname, realpath, isdir, exists
import glob
import multiprocessing
import threading
import argparse
import classifier
from sklearn.mixture import GMM
np.set_printoptions(precision=2)

import openface
threshold = .7
videoSend = "videoOut.avi"
videoReceive = "videoTemp.avi"
faceDir = "faces"
exitCode = False
notEnoughData = True
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizerMutex = False
fileDir = dirname(realpath(__file__))
modelDir = join(fileDir, 'models')
dlibModelDir = join(modelDir, 'dlib')
openfaceModelDir = join(modelDir, 'openface')

align = openface.AlignDlib(classifier.dlibFacePredictor)
net = openface.TorchNeuralNet(classifier.networkModel, classifier.imgDim)

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

def infer(reps):
    with open(classifier.classifierFile, 'r') as f:
        (le, clf) = pickle.load(f)

    persons = []
    confidences = []

    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print "No Face detected"
            return (None, None)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        persons.append(le.inverse_transform(maxI))
        confidences.append(predictions[maxI])
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences)

def detectLoop():
    global frame
    while not exitCode:
        if frame is not None:
            print "Detecting"
            detect(frame)
            connData.send(str([list(rects), list(texts)]))
        else:
            print "Waiting on frame"
def detect(frame):
    global recognizerMutex
    xTemp, yTemp = 0, 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rectsTemp, textTemp = [], []

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get all bounding box es
    bb = align.getAllFaceBoundingBoxes(rgbImg)

    if bb is None:
        return

    alignedFaces = []

    for box in bb:
        rectsTemp.append([(box.left(),box.top()),(box.right(),box.bottom()), (255,255,255),1])
        alignedFaces.append(
            align.align(
                classifier.imgDim,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if alignedFaces is None:
        raise Exception("Unable to align the frame")

    start = time.time()

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    while recognizerMutex:
        print "Waiting on mutex in detect"
        time.sleep(.05)

    recognizerMutex = True
    persons, confs = infer(reps)
    recognizerMutex = False

    print persons, confs

    possibleFaces = []
    for i, (person, conf) in enumerate(zip(persons, confs)):
        if conf>threshold:
            rectsTemp[i][2]=(0,0,255)#Change detected face box to red
            textTemp.append(("%s: %.2f"%(person,conf),(rectsTemp[i][0][0],rectsTemp[i][1][1]+20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255),2,cv2.CV_AA))

    #
    # rectsTemp.append((fromCoord,toCoord, (0,0,255),1))
    # textTemp.append((faceFiles[predicted], fromCoord, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.CV_AA))

    while rects:
        rects.pop()#clear the list

    for r in rectsTemp:
        rects.append(tuple(r))

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
    print faceFiles, templates

def trainNetwork():
    global recognizer, templates, faceFiles, recognizerMutex
    while recognizerMutex:
        print "Waiting on mutex in train"
        time.sleep(.05)
    recognizerMutex = True
    classifier.train()
    recognizerMutex = False

def main():
    global rects, texts, templates, faceFiles, frame
    frame = None
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

    threading.Thread(target=detectLoop).start()

    while True:
        flag, frameTemp = cap.read()
        if flag:
            frame = frameTemp
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            time.sleep(.5)
        if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            break

def dataReceive():
    tempBuffer = ""
    firstPart, secondPart = None, None
    while not exitCode:
        data = connData.recv(2**15)
        tempBuffer+=data
        while boundary in tempBuffer and not exitCode:
            ind = tempBuffer.index(boundary)
            rawPart = tempBuffer[:ind]

            tempBuffer = tempBuffer[ind+len(boundary):]
            if not firstPart:
                print "In first part"
                firstPart = rawPart
            else:
                print "In second part"
                secondPart = rawPart

                #Process it
                img = pickle.loads(firstPart)
                dirname = pickle.loads(secondPart)
                newDir =join("faces/", dirname)
                if not exists(newDir):
                    mkdir(newDir)

                imgNum = 1
                while exists(join(newDir,str(imgNum)+".png")):
                    print("Already exists: "+join(newDir,str(imgNum)+".png"))
                    imgNum+=1

                cv2.imwrite(join(newDir, str(imgNum) + '.png'), img)


                firstPart,secondPart = None, None
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
        try:
            print("In Interrupt")
            exitCode = True
            unlink(videoSend)
            s.close()
            cap.release()
            outputVideo.release()
            cv2.destroyAllWindows()
            print("End of interrupt")
        finally:
            raise
