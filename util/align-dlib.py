#!/usr/bin/env python2
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

import argparse
import cv2
import numpy as np
import os
import random
import shutil

import openface
import openface.helper
from openface.data import iterImgs

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

dlibFacePredictor =os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat" )
landmarks = "outerEyesAndNose"
outputDirectory = "faces/"
size = 96

def write(vals, fName):
    if os.path.isfile(fName):
        print("{} exists. Backing up.".format(fName))
        os.rename(fName, "{}.bak".format(fName))
    with open(fName, 'w') as f:
        for p in vals:
            f.write(",".join(str(x) for x in p))
            f.write("\n")

def align(img, path):
    landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE
    align = openface.AlignDlib(dlibFacePredictor)
    outDir = os.path.join(outputDir, path)
    outputPrefix = os.path.join(outDir, imgObject.name)
    imgName = outputPrefix + ".png"
    outRgb = align.align(args.size, img,
                         landmarkIndices=landmarkIndices,
                         skipMulti=True)
    if outRgb is None and args.verbose:
        print("  + Unable to align.")

    if outRgb is not None:
        if args.verbose:
            print("  + Writing aligned file to disk.")
        outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imgName, outBgr)


if __name__ == '__main__':


    if args.mode == 'computeMean':
        computeMeanMain(args)
    else:
        alignMain(args)
