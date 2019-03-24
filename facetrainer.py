import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

path = 'C:/Users/omkar/Desktop/ImageDataSet/'
onlyfiles = [f for f in listdir(path) if isfile(join(path,f))] 

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete!")