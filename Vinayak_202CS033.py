import Augmentor
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
from sklearn.metrics import confusion_matrix , classification_report
import pandas as pd
import seaborn as sns

#offline data augmentation
p = Augmentor.Pipeline("C:\\Users\\pcvin\\Downloads\\study\\dl\\project\\data\\Train")

p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.3, min_factor=1.01, max_factor=1.1)
p.flip_random(probability=0.5)
p.random_brightness(probability=0.5,min_factor=0.7,max_factor=1.3)
p.random_distortion(probability=0.2, grid_width=3,grid_height=3,magnitude=1)
p.shear(probability=0.3,max_shear_left=10,max_shear_right=10)
p.sample(20000,multi_threaded=True)

#loading data
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
#image normalization
x_train=x_train/255
x_test=x_test/255

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

#build model
base_model = applications.resnet.ResNet101(weights= None, include_top=False, input_shape= (100,100,1))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(2, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
adam = Adam(lr=0.00001)
model.compile(optimizer= adam, loss='binary_crossentropy', metrics=['accuracy'])
#training the model
model.fit(x_train, y_train, epochs = 20, batch_size = 64, shuffle=True)
#save the trained model for future use
model.save('C:\\Users\\pcvin\\Downloads\\study\\dl\\project\\models\\save1')


#load model when needed (running below code in a spereate .py file is adviced)
model = keras.models.load_model('C:\\Users\\pcvin\\Downloads\\study\\dl\\project\\models\\save1')
#predict test data
ypred = model.predict(x_test)
ypred = ypred[:, 0]
ypred=np.round(ypred)
y_test = y_test[:, 0]
cm = confusion_matrix(y_test,ypred)

#printing performace matrices
print(classification_report(y_test,ypred))

#displaying confusion matrix
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')
    plt.show()
print_confusion_matrix(cm,["Covid","Not Covid"])