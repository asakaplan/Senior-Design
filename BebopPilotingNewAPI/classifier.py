#!/usr/bin/env python2
# MODIFIED from original apache license (2017):
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import os
import pickle

from os.path import isdir, join

from operator import itemgetter
import numpy as np
import pandas as pd
import openface
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from random import sample

np.set_printoptions(precision=2)
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


dlibFacePredictor = os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")
networkModel = os.path.join(openfaceModelDir,'nn4.small2.v1.t7')

workDir = "data/"
classifierFile = "{}/classifier_%02d.pkl".format(workDir)
imgDim = 96

#Decision forest parameters
ensembleSize = 60
sampleRatio = .7
minFacesPerPerson = 5
unknownThreshold  = .5
questionableThreshold = .7
trainingRunning = False
needsToRun = False

class Classifier:
    def __init__(self):
        self.trainingRunning = False
        self.needsToRun = False
    def train(self):
        if self.trainingRunning:
            print "Deferring run"
            self.needsToRun = True
            return
        self.trainingRunning = True

        try:
            os.remove("faces/cache.t7")
        except:
            pass
        os.system("batch-represent/main.lua -outDir data/ -data faces/")
        images = "faces/"
        cuda = True

        align = openface.AlignDlib(dlibFacePredictor)
        net = openface.TorchNeuralNet(networkModel, imgDim=imgDim, cuda=cuda)
        workDir="data/"
        fname = "{}/labels.csv".format(workDir)
        labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
        labels = map(itemgetter(1),
                     map(os.path.split,
                         map(os.path.dirname, labels)))  # Get the directory.
        fname = "{}/reps.csv".format(workDir)
        embeddings = pd.read_csv(fname, header=None).as_matrix()
        keyPairs = {}
        for label, embedding in zip(labels, embeddings):
            if label in keyPairs:
                keyPairs[label].append(embedding)
            else:
                keyPairs[label]=[embedding]

        for i in range(1,1+ensembleSize):
            subSample = {label:sample(embeddings,min(len(embeddings),minFacesPerPerson,int(round(sampleRatio*len(embeddings))))) for label, embeddings in keyPairs.items()}
            labelsSample, embeddingsSample = [], []
            for label, embeddings in subSample.items():
                for embedding in embeddings:
                    labelsSample.append(label)
                    embeddingsSample.append(embedding)

            le = LabelEncoder().fit(labelsSample)
            labelsNum = le.transform(labelsSample)
            nClasses = len(le.classes_)
            clf = SVC(C=1, kernel='linear', probability=True)#GaussianNB()#(C=1, kernel='linear', probability=True)
            clf.fit(embeddingsSample, labelsNum)
            with open(classifierFile%i, 'w') as f:
                pickle.dump((le, clf), f)
        print "Setting trainingRunning to false", self.trainingRunning, self.needsToRun
        self.trainingRunning = False
        if self.needsToRun:
            print "Rerunning train from defer"
            train()
            needsToRun=False
    def infer(self, reps):
        listOfResults = [{} for i in reps]
        totalPredicted = [0 for i in reps]
        for i in range(1,1+ensembleSize):
            with open(classifierFile%i, 'r') as f:
                (le, clf) = pickle.load(f)

            persons = []
            confidences = []

            for ind, rep in enumerate(reps):
                try:
                    rep = rep.reshape(1, -1)
                except:
                    print "One of the detected faces is invalid"
                    continue
                predictions = clf.predict_proba(rep).ravel()
                maxI = np.argmax(predictions)
                name = le.inverse_transform(maxI)
                if predictions[maxI]>1.0/(len(predictions)-1):
                    listOfResults[ind][name] = 1+ listOfResults[ind].get(name,0)
                    totalPredicted[ind]+=1

        for i in range(len(reps)):
            selected = max(listOfResults[i].items(),key=lambda a:a[1])
            confidence = float(selected[1])/totalPredicted[i]
            persons.append(selected[0])
            confidences.append(confidence)

        return (persons, confidences)
