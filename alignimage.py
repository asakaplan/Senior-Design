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

# Modified significantly from source

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
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

dlibFacePredictor =os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat" )
landmarks = "outerEyesAndNose"
size = 96


def align(img, outDir):
    """Aligns an image with its eyes and outputs to outDir"""
    landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE
    align = openface.AlignDlib(dlibFacePredictor)
    outRgb = align.align(size, img,
                         landmarkIndices=landmarkIndices,
                         skipMulti=True)
    if outRgb is None:
        print("  + Unable to align.")

    if outRgb is not None:
        outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
        print("Creating new image:",outDir)
        cv2.imwrite(outDir, outBgr)
    else:
        print "Nothing outputted by align"
