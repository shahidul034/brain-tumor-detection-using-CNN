import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import  MaxPooling2D
from keras.layers import Flatten, Dense
import cv2

from keras_preprocessing import image

img_width, img_height = 64, 64

def create_model():
    model = Sequential()


    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())


    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=3, activation='sigmoid'))
    model.summary()
    return model


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory(r'C:\Users\User\Desktop\data\train',
                                                 target_size=(64, 64),
                                                 batch_size=64,
                                                 class_mode='categorical')


##################################
from keras.preprocessing import image
from keras.models import load_model
model = load_model('currnencymodel.h5')
classifier = create_model()

test_image = image.load_img(r'C:\Users\User\Desktop\data\predict\L1\test857.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#predict the result
result = classifier.predict(test_image)

print((result))

arr=np.array(result).tolist()
#print(arr)
ch=0
cnt=0
ans=1
#for x in arr:
#    ch=ch+1
 #   print(type(x))
 #   if cnt >= int(x):
#        ans = ch
 #       cnt=x
#print("ans: ",ans)









########################
