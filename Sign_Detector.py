# %%  Import libraries

from keras.preprocessing import image
from keras.models import load_model

import numpy as np
import cv2 as cv

# %%  Load model

model = load_model('traffic_sign_recognitor.model')

# %%   Camera resolution

frame_width = 900
frame_height = 600
font = cv.FONT_HERSHEY_PLAIN

# %%  Webcam

webcam = cv.VideoCapture(0)
webcam.set(3, frame_width)
webcam.set(4, frame_height)

# %% labels name

label_names = {
    0 : 'Speed limit 20km/h', 
    1 : 'Speed limit 30km/h',
    2 : 'Speed limit 50km/h',
    3 : 'Speed limit 60km/h',
    4 : 'Speed limit 70km/h' , 
    5 : 'Speed limit 80km/h',
    6 : 'End of speed limit 80km/h' ,
    7 : 'Speed limit 100km/h' , 
    8 : 'Speed limit 120km/h' , 
    9 : 'No passing',
    10 : 'No passing for vehicles over 3.5 metric tons',
    11 : 'Right-of-way at the next intersection',
    12 : 'Priority road',
    13 : 'Yield' ,
    14 : 'Stop' ,
    15 : 'No vehicles',
    16 : 'Vehicles over 3.5 metric tons prohibited' ,
    17 : 'No entry',
    18 : 'General caution' ,
    19 : 'Dangerous curve to the left',
    20 : 'Dangerous curve to the right' ,
    21 : 'Double curve',
    22 : 'dumpy road' ,
    23 : 'Slippery road',
    24 : 'Road narrows on the right' ,
    25 : 'Road work',
    26 : 'Traffic signals' ,
    27 : 'Pedestrians' ,
    28 : 'Children crossing',
    29 : 'cycles crossing' ,
    30 : 'aware of ice/snow',
    31 : 'Wild animals crossing',
    32 : 'End of all speed and passing limits' ,
    33 : 'Turn right ahead',
    34 : 'Turn left ahead' ,
    35 : 'Ahead only' ,
    36 : 'Go straight or right',
    37 : 'Go straight or left' ,
    38 : 'Keep right' ,
    39 : 'Keep left',
    40 : 'Roundabout mandatory',
    41 : 'End of no passing',
    42 : 'End of no passing y vehicles over 3.5 metric tons'
}

labels = list(label_names.values())


# %%   Functions

def grayscale(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv.equalizeHist(img)
    return img

def preprocessing(img):
    # img = grayscale(img)
    # img = equalize(img)
    img = image.img_to_array(img)
    img = img.astype(np.float32)/255.0  
    img = np.expand_dims(img, axis=0)
    return img

def getClassName(classNo):
    return labels[classNo]

# %%  loop through the frames

while webcam.isOpened():

    # read frames through the webcam
    status, frame = webcam.read()

    # process image ; detect object region 
    img = cv.resize(frame, (32,32))
    img = preprocessing(img)

    # write label above the object
    cv.putText(frame, 'CLASS: ', (20,35), font, 1.6, (255,10,0), 2)

    # predict the image
    pred = model.predict(img)

    # get the model with max accuracy
    index = np.argmax(pred)
    label = labels[index]

    cv.putText(frame, label, (120,35), font, 1.6, (10,100,150), 2)
    cv.imshow('Traffic Sign Recognition', frame)


    if cv.waitKey(1) & 0xff == ord('q'):
        break

# release resources
webcam.release()
cv.destroyAllWindows()

