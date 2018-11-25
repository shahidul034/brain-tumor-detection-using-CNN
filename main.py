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
