import time

start = time.time()

import cv2
import numpy as np
import sys
import Tkinter as tk
from os import listdir
from os.path import isfile, join, dirname, realpath
import glob
import multiprocessing
import threading
import argparse
from sklearn.mixture import GMM
np.set_printoptions(precision=2)

import openface

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
    global rects, frame, templates, faceFiles
    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, ((x,y),(x2,y2),_,__) in enumerate(rects):
            if (x < ix) and (x2> ix) and (y < iy) and (y2 > iy):
                cv2.imshow('Recognized', frame[y:y2,x:x2])
                ix, iy = -1, -1
                create_new_text_window()
                cv2.imwrite('faces/' + templateName + '.png', frame[y:y2,x:x2])
                templates= [cv2.imread('faces/' + templateName + '.png', 0)]+templates
                faceFiles =[templateName+".png"]+faceFiles
                break

def detect():
    xTemp = 0
    yTemp = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rectsTemp = []
    textTemp = []
    # Match a Template
    inner = 0
    if frame is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    # Get the largest face bounding box
    # bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box

    # Get all bounding boxes
    bb = align.getAllFaceBoundingBoxes(rgbImg)

    if bb is None:
        # raise Exception("Unable to find a face: {}".format(imgPath))
        return None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()

    alignedFaces = []
    for box in bb:
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

    start = time.time()
    rep = net.forward(alignedFace)
    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        print("Representation:")
        print(rep)
        print("-----\n")
    print(rep)
    print(reps)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
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
def keepChecking():
    while not exitCode:
        if frame!=None:detect()
def infer(img, args):
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)  # le - label and clf - classifer

    reps = getRep(img)
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
        # print predictions
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        # print str(le.inverse_transform(max2)) + ": "+str( predictions [max2])
        # ^ prints the second prediction
        confidences.append(predictions[maxI])
        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
            pass
        # print("Predict {} with {:.2f} confidence.".format(person, confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences)
def execute_main_loop():
    global ix, iy, ievent, master, templateName, rects, texts, curFrame, templates, faceFiles, exitCode, frame
    frame = None
    exitCode = False
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

    sizes = [140, 160, 180, 200, 220, 240, 260]
    xTemp = 0
    yTemp = 0
    threshold = 0.60
    video_capture = cv2.VideoCapture(0)
    frameSkipper = 10
    frameIndex = 0
    process = None
    threading.Thread(target=keepChecking).start()
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        for rect in rects:
            cv2.rectangle(frame,*(rect))

        for text in texts:
            cv2.putText(frame,*(text))
        # Display the resulting frame
        cv2.imshow('Video', frame)

        #Quit when q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exitCode=True
            break

if __name__ == "__main__":
    #global variables
    template = 0
    sizes = [80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
    threshold = 0.8
    ix, iy = -1, -1
    ievent = 0
    templateName = ''
    master = 0
    e = 0
parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument(
        '--captureDevice',
        type=int,
        default=0,
        help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda=args.cuda)
    #video feed source and windows
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow('Video')
    cv2.namedWindow('Recognized')

    cv2.setMouseCallback('Video', get_mouse_position_onclick)
    execute_main_loop()
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
