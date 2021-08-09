import numpy as np
import cv2
import os
import matplotlib.pyplot  as plt
import keras
from keras import applications
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.optimizers import SGD, Adam

x_train = []
y_train = []
x_test = []
y_test = []
def load_images_from_folder(folder,y,trainortest):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE )
        if img is not None:
            if trainortest:
                x_train.append(img)
                y_train.append(y)
            else:
                x_test.append(img)
                y_test.append(y)

load_images_from_folder('C:\\Users\\pcvin\\Downloads\\study\\dl\\project\\data\\Train\\output\\COVID',1,True)
load_images_from_folder('C:\\Users\\pcvin\\Downloads\\study\\dl\\project\\data\\Train\\output\\non-COVID',0,True)
load_images_from_folder('C:\\Users\\pcvin\\Downloads\\study\\dl\\project\\data\\Test\\COVID',1,False)
load_images_from_folder('C:\\Users\\pcvin\\Downloads\\study\\dl\\project\\data\\Test\\non-COVID',0,False)
x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)
x_train=x_train/255
x_test=x_test/255
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

base_model = applications.resnet.ResNet101(weights= None, include_top=False, input_shape= (100,100,1))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(2, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

# cv2.imshow('im;', X[0])
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 
# fig = plt.figure(figsize=(100, 100))  # width, height in inches

# for i in range(100):
#     sub = fig.add_subplot(25, 25, i + 1)
#     sub.imshow(X[i,:,:],cmap='gray', interpolation='nearest')
# plt.show()

adam = Adam(lr=0.00001)
model.compile(optimizer= adam, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 20, batch_size = 64, shuffle=True)
model.save('C:\\Users\\pcvin\\Downloads\\study\\dl\\project\\models\\save1')
preds = model.evaluate(x_test, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))