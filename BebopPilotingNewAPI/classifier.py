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
from sklearn.svm import SVC

np.set_printoptions(precision=2)
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


dlibFacePredictor = os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")
networkModel = os.path.join(openfaceModelDir,'nn4.small2.v1.t7')

workDir = "data/"
classifierFile = "{}/classifier.pkl".format(workDir)
imgDim = 96

def train():
    os.remove("faces/cache.t7")
    os.system("batch-represent/main.lua -outDir data/ -data faces/")
    images = "faces/"
    cuda = True

    align = openface.AlignDlib(dlibFacePredictor)
    net = openface.TorchNeuralNet(networkModel, imgDim=imgDim, cuda=cuda)
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(workDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(workDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))
    clf = SVC(C=1, kernel='linear', probability=True)
    print(len(embeddings),len(labelsNum))
    clf.fit(embeddings, labelsNum)

    print("Saving classifier to '{}'".format(classifierFile))
    with open(classifierFile, 'w') as f:
        pickle.dump((le, clf), f)
