import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'   
)

#테스트 데이터의 목적은 평가이니, 증폭하지 않은 원데이터를 쓴다. 정확한 평가를 위해서 증폭을 할 필요가 없음.                  
test_datagen = ImageDataGenerator(
    rescale=1./255
)                           

#(x= 80, 150, 150, 1)인 부분이  x = (160, 150, 150, 1) y =(160, )  
xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',
    target_size=(100, 100), #줄어듬 기존 사이즈가 150,150이니깐
    batch_size=100,          #배치를 크게 잡아주면 총 데이터 수를 알 수 있게 된다. 
    class_mode='binary',
    color_mode='grayscale', #끝자리가 1이 되는 이유
    shuffle=True
    # Found 160 images belonging to 2 classes.
)


xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size=(100, 100), #줄어듬 기존 사이즈가 150,150이니깐
    batch_size=100,
    class_mode='binary',
    color_mode='grayscale', #끝자리가 1이 되는 이유
    shuffle=True
    # Found 120 images belonging to 2 classes.
)


print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000171BF38CD90>


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(100, 100, 1)))
model.add(Conv2D(80, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(80, (3,3), activation='relu'))
model.add(Conv2D(100, (3,3), activation='relu'))
model.add(Conv2D(50, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(38, activation='relu'))
model.add(Dense(73, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

# hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=100, 
#                     validation_data=xy_test, 
#                     validation_steps=4, )   
hist = model.fit(#xy_train[0][0], xy_train[0][1], 
                 xy_train,
                 #steps_per_epoch=16, 
                 epochs=100, 
                 validation_data=(xy_test[0][0], xy_test[0][1]))
                    # validation_steps=4, 
                    #validation_split=0.2)   

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss :', loss[-1])    
print('val_loss :', val_loss[-1])   
print('accuracy :', accuracy[-1])
print('val_acc :', val_acc[-1])
   
# import matplotlib.pyplot as plt
# plt.imshow(xy_train[0][0][1],'gray')
# plt.show()


#4. 결과 

