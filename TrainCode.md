```python
import numpy as np
import cv2
import glob
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split


#This method reads all the images in the folder, converts them to grayscale, and returns them in a list.
def get_image(path):
    images = []
    for file in glob.glob(path):
        img = cv2.imread(file,0)
        images.append(img)
    return images
    
#This method blurs the image by creating a kernel of size 4x4.
def blur(images):
    newImages=[]
    window=np.ones((4,4),np.float32)/16
    for img in images:
        temp=cv2.filter2D(img,-1,window)
        newImages.append(temp)
    return newImages
    
#This method resizes the images to 20% of its original height and width.
def simplify(images):
    newImages=[]
    for img in images:
        temp = cv2.resize(images[10],(0,0),fx=0.2,fy=0.2)
        newImages.append(temp)
    return newImages
    
#Loading the data
rock = simplify(blur(get_image('G:/path/rock/*.*')))
paper = simplify(blur(get_image('G:/path/paper/*.*')))
scissors = simplify(blur(get_image('G:/path/scissors/*.*')))

#creating dataset
data=[]
target=[]
for i in rock:
    data.append(i)
    target.append(0)
for i in paper:
    data.append(i)
    target.append(1)
for i in scissors:
    data.append(i)
    target.append(2)
    
#Splitting the data into train and test sets.
x_train,x_test,y_train,y_test=train_test_split(data,target)

x_train=np.array(x_train)                         #The data is in the form of a list, which needs to be converted to a numpy array.
x_test=np.array(x_test)
x_train=x_train.reshape((1641,2400))              #The images are of dimension 40x60.
x_test=x_test.reshape((547,2400))
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
y_train=keras.utils.to_categorical(y_train,3)
y_test=keras.utils.to_categorical(y_test,3)

#Creating model.
model=Sequential()
model.add(Dense(1024,input_shape=(2400,),activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dense(3,activation='softmax'))

#Compiling and training the model.
model.compile(optimizer=RMSprop(),loss='categorical_crossentropy',metrics=['accuracy'])
h=model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test))
model.save('G:/path/train_model.h5')

```
