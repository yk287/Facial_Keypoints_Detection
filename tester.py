import numpy as np
import matplotlib.pyplot as plt

import cv2
image = cv2.imread('images/face.jpeg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(9,9))
plt.imshow(image)

face_cascade = cv2.CascadeClassifier('/home/youngwook/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(image, 1.3, 5)

image_with_detections = image.copy()

for (x,y,w,h) in faces:

    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)

fig = plt.figure(figsize=(9,9))

plt.imshow(image_with_detections)

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from models import Net
net = Net(1).to(device)
net.load_state_dict(torch.load('saved_models/final_model_augmented_300epoch_no_batchnorm.pt'))

net.eval()


image_copy = np.copy(image)
import cv2

# loop over the detected faces from your haar cascade
for (x, y, w, h) in faces:
    # Select the region of interest that is the face in the image

    roi = image_copy[y:y + h, x:x + w]
    print(roi.shape)
    ## TODO: Convert the face region from RGB to grayscale

    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi = roi / 255.0
    ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    roi = cv2.resize(roi, (224, 224))
    ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)

    # roi = roi.reshape(roi.shape[0], roi.shape[1], 1)
    # roi = roi.transpose((2, 0, 1))

    # this part is a little different from the way it's done in notebook 2, but it does what it's supposed to
    roi_input = torch.from_numpy(roi).float().unsqueeze(0).unsqueeze(0).to(device)
    ## TODO: Make facial keypoint predictions using your loaded, trained network

    prediction = net(roi_input)

    prediction = prediction.view(prediction.size()[0], 68, -1)

    visualize_output(roi_input, prediction, batch_size=1)
