# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import LeNet
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os
from PIL import Image


classNames=['V0_C0', 'V0_C1', 'V0_C2', 'V0_C3', 'V0_C4', 'V0_C5', 'V0_C6', 'V0_C7', 'V0_C8', 'V0_C9', 'V1_C0', 'V1_C1', 'V1_C2', 'V1_C3', 'V1_C4', 'V1_C5', 'V1_C6', 'V1_C7', 'V1_C8', 'V1_C9', 'V2_C0',
 'V2_C1', 'V2_C2', 'V2_C3', 'V2_C4', 'V2_C5', 'V2_C6', 'V2_C7', 'V2_C8', 'V2_C9', 'V3_C0', 'V3_C1', 'V3_C2', 'V3_C3', 'V3_C4', 'V3_C5', 'V3_C6', 'V3_C7', 'V3_C8', 'V3_C9', 'V4_C0', 'V4_C1',
 'V4_C2', 'V4_C3', 'V4_C4', 'V4_C5', 'V4_C6', 'V4_C7', 'V4_C8', 'V4_C9', 'V5_C0', 'V5_C1', 'V5_C2', 'V5_C3', 'V5_C4', 'V5_C5', 'V5_C6', 'V5_C7', 'V5_C8', 'V5_C9', 'V6_C0', 'V6_C1', 'V6_C2',
 'V6_C3', 'V6_C4', 'V6_C5', 'V6_C6', 'V6_C7', 'V6_C8', 'V6_C9', 'V7_C0', 'V7_C1', 'V7_C2', 'V7_C3', 'V7_C4', 'V7_C5', 'V7_C6', 'V7_C7', 'V7_C8', 'V7_C9', 'V8_C0', 'V8_C1', 'V8_C2', 'V8_C3',
 'V8_C4', 'V8_C5', 'V8_C6', 'V8_C7', 'V8_C8', 'V8_C9', 'V9_C0', 'V9_C1', 'V9_C2', 'V9_C3', 'V9_C4', 'V9_C5', 'V9_C6', 'V9_C7', 'V9_C8', 'V9_C9']

translate=['ஃ','க்']



im = Image.open('106.png').convert('RGB')
im2arr = np.array(im) # im2arr.shape: height x width x channel
img=np.array(im)
print(im2arr.shape)

# # In case of grayScale images the len(img.shape) == 2
# if len(im2arr.shape) > 2 and im2arr.shape[2] == 4:
#     #convert the image from RGBA2RGB
#     im2arr= cv2.cvtColor(im2arr, cv2.COLOR_BGRA2BGR)
#print(im2arr.shape)
im2arr = np.expand_dims(im2arr, axis=0)
print(im2arr.shape)

model = load_model('./resnet_checkpoint/checkpoint.hdf5')
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(im2arr)
print(type(predictions))
print(predictions.argmax(axis=1))
print(classNames[predictions.argmax(axis=1)[0]])
#
# plt.imshow(img)
# plt.title(translate[predictions.argmax(axis=1)[0]])
# plt.show()


