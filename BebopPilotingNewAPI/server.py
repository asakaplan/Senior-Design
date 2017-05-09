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
import alignimage
np.set_printoptions(precision=2)

import openface
threshold = .7
videoSend = "videoOut.avi"
videoReceive = "videoTemp.avi"
faceDir = "faces"
exitCode = False
notEnoughData = True
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizerMutex = []
fileDir = dirname(realpath(__file__))
modelDir = join(fileDir, 'models')
dlibModelDir = join(modelDir, 'dlib')
openfaceModelDir = join(modelDir, 'openface')

align = openface.AlignDlib(classifier.dlibFacePredictor)
net = openface.TorchNeuralNet(classifier.networkModel, classifier.imgDim)



def setupSocket(port):
    """Starts a process to listen for a socket connection on port number
       'port', with blocking. Returns the socket and connection when a
       connection is established"""
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
    """Returns true if the argument has a boolean function of True and when
    the value is not nan"""
    return bool(val) and not isnan(val)


def detectLoop():
    """Synchronous loop that continually runs detect when a frame is available"""
    global frame, cf
    cf = classifier.Classifier()

    while not exitCode:
        if frame is not None:
            detect(frame)
            connData.send(str([list(rects), list(texts)]))
        else:
            print "Waiting on frame"

def detect(frame):
    """Detects and names faces in the given frame, outputting to the global
    variables rects and texts"""

    global recognizerMutex, cf
    xTemp, yTemp = 0, 0
    rectsTemp, textTemp = [], []

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
        #print "Waiting on mutex in detect"
        time.sleep(.5)

    persons, confs = cf.infer(reps,recognizerMutex)
    recognizerMutex = []

    print persons, confs

    possibleFaces = []
    for i, (person, conf) in enumerate(zip(persons, confs)):
        if "unknown" not in person:
            if conf>classifier.questionableThreshold:
                rectsTemp[i][2]=(0,0,255)#Change detected face box to red
                textTemp.append(("%s: %.2f"%(person,conf),(rectsTemp[i][0][0],rectsTemp[i][1][1]+20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255),2,cv2.CV_AA))
            elif conf<classifier.unknownThreshold:
                rectsTemp[i][2]=(255,255,255)#Change detected face box to red
                textTemp.append(("(Very Unknown): %.2f"%conf,(rectsTemp[i][0][0],rectsTemp[i][1][1]+20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255),2,cv2.CV_AA))
            else:
                textTemp.append(("(Questionable %s): %.2f"%(person,conf),(rectsTemp[i][0][0],rectsTemp[i][1][1]+20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,0),2,cv2.CV_AA))
        else:
            textTemp.append(("(Very unknown)",(rectsTemp[i][0][0],rectsTemp[i][1][1]+20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,0),2,cv2.CV_AA))

    while rects:
        rects.pop()#clear the list

    for r in rectsTemp:
        rects.append(tuple(r))

    while texts:
        texts.pop()#clear the list

    for t in textTemp:
        texts.append(t)

def setupFiles():
    """Creates the needed fifo files and sets them to have a larger buffer size
    via a perl call"""
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

def trainNetwork():
    """Trains the neural network as soon as it is available, based on the mutex"""
    global recognizerMutex, cf
    while recognizerMutex:
        print "Waiting on mutex in train"
        time.sleep(.5)
    cf.train(recognizerMutex)

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

    threading.Thread(target=detectLoop).start()
    threading.Thread(target=trainNetwork).start()

    while not exitCode:
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
    """Receives click data from the data socket connection"""
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
                    imgNum+=1
                alignimage.align(img, join(newDir, str(imgNum) + '.png'))


                firstPart,secondPart = None, None
                threading.Thread(target=trainNetwork).start()

    tempFile.close()

def videoDataReceive():
    """Receives video stream from socket connection and outputs it to tempFile
    pipe. Also sends video directly back to the video connection socket"""
    tempFile = open(videoReceive,"wb")
    while not exitCode:
        data = conn.recv(2**15)
        tempFile.write(data)
        connVideo.send(data)
    tempFile.close()

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
