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
import Tkinter as tk
from os import listdir
from os.path import isfile, join, dirname, realpath
import glob
import multiprocessing
import threading
import argparse

np.set_printoptions(precision=2)

import openface

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
parser = argparse.ArgumentParser()

parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))


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
    global le, clf

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
    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))


    rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the largest face bounding box
    # bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box

    # Get all bounding box es
    bb = align.getAllFaceBoundingBoxes(rgbImg)

    if bb is None:
        # raise Exception("Unable to find a face: {}".format(imgPath))
        return None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))
    alignedFaces = []
    for box in bb:
        rectsTemp.append(((box.left(),box.top()),(box.right(),box.bottom()), (255,255,255),1))
        alignedFaces.append(
            align.align(
                args.imgDim,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    print("Reps:", reps)
    persons, confs = infer(reps)
    print persons, confs
    #
    # while recognizerMutex:
    #     print "Waiting on mutex in detect"
    #     time.sleep(.05)
    # recognizerMutex = True
    possibleFaces = []
    for face, fromCoord, toCoord in possibleFaces:
        faceGrey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        predicted, conf = recognizer.predict(faceGrey)
        if conf>threshold:
            rectsTemp.append((fromCoord,toCoord, (0,0,255),1))
            textTemp.append((faceFiles[predicted], fromCoord, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.CV_AA))
        else:
            rectsTemp.append((fromCoord,toCoord, (255,255,255),1))
    recognizerMutex = False
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
    print faceFiles, templates

def trainNetwork():
    global recognizer, templates, faceFiles, recognizerMutex
    while recognizerMutex:
        print "Waiting on mutex in train"
        time.sleep(.05)
    recognizerMutex = True
    recognizer = cv2.createLBPHFaceRecognizer()
    nameMap = {}

    for ind, fileThing in enumerate(faceFiles):
        if " " in fileThing:continue
        inda = fileThing.index(".")
        baseName = fileThing[:inda]
        nameMap[baseName]=ind
    res = [0 for i in range(len(faceFiles))]
    for ind, fileThing in enumerate(faceFiles):
        if " " not in fileThing:
            res[ind]=ind
            continue
        inda = fileThing.index(" ")
        baseName = fileThing[:inda]
        res[ind]=nameMap[baseName]
        #check for errors
    recognizer.train(templates, np.array(res))
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
                fileName = pickle.loads(secondPart)
                dirname, imagenum = fileName.split(" ")
                cv2.imwrite('faces/' + dirname + '/' + imagenum + '.png', img)
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
