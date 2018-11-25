# brain-tumor-detection-using-CNN
## main.py
It is the training code.we use keras to train our system
```
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

print("hello")
classifier = Sequential()


classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))


classifier.add(MaxPooling2D(pool_size=(2, 2)))


classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


classifier.add(Flatten())


classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=3, activation='sigmoid'))


classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(r'C:\Users\ARANYA\Desktop\data\train',
                                                 target_size=(64, 64),
                                                 batch_size=64,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(r'C:\Users\ARANYA\Desktop\data\test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch=200,
                         epochs=30,
                         validation_data=test_set,
                         validation_steps = 715 / 64 )
classifier.save('tumormodel.h5');
print("finish")
```

## main2.py
we predict our image using this code
```
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
```
## main3.py
we extract our image file from (.mat file).(.mat) file contains label,patient ID,image data
```
import h5py
import cv2
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)

files={}
array = {}
cnt=0
for i in range(2299,3065):
    filepath=r"C:\Users\User\Desktop\data\brainTumorDataPublic_2299-3064/" + str(i) + '.mat'
    f=h5py.File(filepath)
    x=f['cjdata']
    img=np.array(x['image'])
    #type(img)
    #img
    #cv2.imwrite('test'+str(i)+'.jpg',img)

    array['img']=np.array(x['image'])
    array['label']=int(np.array(x['label'])[0][0])
    array['tumormask'] = np.array(x['tumorMask'])
    if array['label'] == 1 :
        cv2.imwrite(r"C:\Users\User\Desktop\data\L1/" + 'test' + str(i) + '.jpg', img)
    if array['label'] == 2 :
        cv2.imwrite(r"C:\Users\User\Desktop\data\L2/" + 'test' + str(i) + '.jpg', img)
    if array['label'] == 3 :
        cv2.imwrite(r"C:\Users\User\Desktop\data\L3/" + 'test' + str(i) + '.jpg', img)
    files[i]=array
```
This brain tumor dataset containing 3064 T1-weighted contrast-inhanced images
from 233 patients with three kinds of brain tumor: meningioma (708 slices), 
glioma (1426 slices), and pituitary tumor (930 slices). Due to the file size
limit of repository, we split the whole dataset into 4 subsets, and achive 
them in 4 .zip files with each .zip file containing 766 slices.The 5-fold
cross-validation indices are also provided.

-----
This data is organized in matlab data format (.mat file). Each file stores a struct
containing the following fields for an image:

cjdata.label: 1 for meningioma, 2 for glioma, 3 for pituitary tumor
cjdata.PID: patient ID
cjdata.image: image data
cjdata.tumorBorder: a vector storing the coordinates of discrete points on tumor border.
		For example, [x1, y1, x2, y2,...] in which x1, y1 are planar coordinates on tumor border.
		It was generated by manually delineating the tumor border. So we can use it to generate
		binary image of tumor mask.
cjdata.tumorMask: a binary image with 1s indicating tumor region
## [brain tumor dataset](https://figshare.com/articles/brain_tumor_dataset/1512427)
