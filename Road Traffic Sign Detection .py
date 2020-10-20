
# %% Importing libraries

import pickle
import numpy as np
import matplotlib.pyplot as plt

# %%  Extracting train data

train_file = 'train.p'
train_obj = open(train_file, 'rb')
train_data = pickle.load(train_obj)
# train_data

# %%  Extracting test data

test_file = 'test.p'
test_obj = open(test_file, 'rb')
test_data = pickle.load(test_obj)
# test_data

# %%  Extracting valid data

valid_file = 'valid.p'
valid_obj = open(valid_file, 'rb')
valid_data = pickle.load(valid_obj)
# valid_data

# %%  Keys

train_data.keys()
# %%  Extracting training feature and training labels

train_features = train_data['features']
train_labels = train_data['labels']

# %%  Extracting testing feature and testing labels

test_features = test_data['features']
test_labels = test_data['labels']

# %%  Extracting valid feature and testing labels

valid_features = valid_data['features']
valid_labels = valid_data['labels']

# %%  Verify the data


plt.figure(figsize=(20,20))
for i in range(50):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.axis('off')
    plt.imshow(test_features[i])
    plt.xlabel(test_labels[i])
plt.show()

# %%  Converting into float 

x_train = train_features.astype('float32')
x_test = test_features.astype('float32')
x_valid = valid_features.astype('float32')

# %%  Normalizing 0 to 1

x_train /= 255
x_test /= 255
x_valid /= 255 

# %% Convert class metrix to binary metrix 

y_train = train_labels
y_test = test_labels
y_valid = valid_labels

# %%   Import TensorFlow

import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Input ,Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing import image

# %%   Create the convolutional base

input_x = Input(shape=(32,32,3))
layer_1 = Conv2D(32, kernel_size=(3,3), activation='relu') (input_x)
layer_2 = MaxPooling2D((2, 2))(layer_1)

layer_3 = Dropout(0.5) (layer_2) # To prevent overfitting

layer_4 = Flatten() (layer_3)
layer_5 = Dense(150, activation='relu') (layer_4)
layer_6 = Dense(80, activation='relu') (layer_5)
layer_7 = Dense(43, activation='softmax') (layer_6)

# %%  Displaying the architecture of model

model = Model(input_x, layer_7)
# model.summary()


# %%  Compile and train the model

model.compile(
    optimizer='Adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )


Epochs=10
BatchSize=200


process = model.fit(
    x_train,
    y_train, 
    epochs= Epochs,
    batch_size= BatchSize,
    validation_data= (x_valid, y_valid ),
    # steps_per_epoch= len(x_train)//BatchSize,
    # validation_data=(valid_test, y_test )
    shuffle=True
    )


# %%  Loss Visualization


plt.style.use('ggplot')
plt.figure(2)
plt.plot(process.history['loss'],label='training loss')
plt.plot(process.history['val_loss'],label='testing loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')

plt.savefig('../Loss.png')

# %%  Accuracy Visualization

plt.style.use('ggplot')
plt.figure(2)
plt.plot(process.history['accuracy'], label='training accuracy')
plt.plot(process.history['val_accuracy'], label='testing accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')

plt.savefig('../Accuracy.png')

# %%  Evaluate the score 

score = model.evaluate(x_test, y_test, verbose=0)

print('loss=', score[0]*100)
print('accuracy=',score[1]*100)


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

labels =label_names.values()
val = list(labels)


# %%  single value prediction

number = 200

prediction = model.predict(x_valid)
warning = np.argmax(prediction[number])
print(val[warning])

plt.axis('off')
plt.imshow(x_valid[number], cmap = plt.cm.binary)
plt.show()
  
# %%  Save the model to the disk

model.save('../traffic_sign_recognitor.model')

# %%

