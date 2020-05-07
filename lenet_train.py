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
import argparse
import os


ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,
                help="path to input dataset")
args=vars(ap.parse_args())

# grab the list of images that we’ll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-1][:5] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
print(classNames)


# initialize the image preprocessors
#aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
 # to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY)=train_test_split(data,labels,test_size=0.25,random_state=42)

trainY=LabelBinarizer().fit_transform(trainY)
testY=LabelBinarizer().fit_transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.05)
model = LeNet.build(width=64, height=64, depth=3,
classes=len(classNames))

# model.compile(loss="categorical_crossentropy", optimizer=opt,
# metrics=["accuracy"])
#
# # # Construct the callback to save only the 'best' model to disk based on the validation loss
# checkpoint = ModelCheckpoint('./checkpoint_dataagu/checkpoint.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)
# callbacks = [checkpoint]
#
#
# # train the network
# print("[INFO] training network...")
# #H= model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=32, epochs=30,callbacks=callbacks, verbose=1)
# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
# epochs=100,callbacks=callbacks,verbose=1)

model = load_model('./checkpoint_dataagu/checkpoint.hdf5')

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=classNames))
print(classNames)

# # plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.show()
