import numpy as np
np.random.seed(1337) 
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

training_set = train_datagen.flow_from_directory(r'C:\Users\ARANYA\Desktop\data\train',
                                                 target_size=(64, 64),
                                                 batch_size=64,
                                                 class_mode='categorical')


##################################
from keras.preprocessing import image
from keras.models import load_model
model = load_model('tumormodel.h5')
classifier = create_model()

test_image = image.load_img(r'C:\Users\ARANYA\Desktop\data\predict\m1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)

#print(result)
#ans=np.array(result).tolist()
maxi=10000.0
ch=0
ans2=1
b = result.ravel()
print(b)



for x in b :
    
    ch=ch+1 
    if (float(1.0-x)<=maxi):
        
        maxi=float(1.0-x)
        print("maxi: ",maxi)
        ans2=ch
        
print("Ans: ",ans2)        
print("Finished")
