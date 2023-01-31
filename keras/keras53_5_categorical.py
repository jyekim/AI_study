import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    rescale= 1./255
)                           

#(x= 80, 150, 150, 1)인 부분이  x = (160, 150, 150, 1) y =(160, )  
xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',
    target_size=(200, 200), #증폭됨 기존 사이즈가 150,150이니깐
    batch_size=10,       #배치를 크게 잡아주면 총 데이터수를 알 수 있게 된다. 
    # class_mode='binary',
    class_mode='categorical',   # binary가 아닌 categoricaldms 원핫이 된것. 
    color_mode='grayscale', #끝자리가 1이 되는 이유
    shuffle=True
    # Found 160 images belonging to 2 classes.
)


xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size=(200, 200), #증폭됨 기존 사이즈가 150,150이니깐
    batch_size=10,
    class_mode='categorical',
    color_mode='grayscale', #끝자리가 1이 되는 이유
    shuffle=True
    # Found 120 images belonging to 2 classes.
)


print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000171BF38CD90>




# print(xy_train[0])             #전체데이터 160개 중에 xy 0번째는 (10, 200, 200, 1)
# print(xy_train[0][0])
# print(xy_train[0][0].shape)   #(10, 200, 200, 1)  여기서 10는 batch_size임
print(xy_train[0][1])            
print(xy_train[0][1].shape)     #(10,2)
# print(xy_train[15][0].shape)    #(10, 200, 200, 1)
# print(xy_train[15][1].shape)    #(10,)

# print(type(xy_train))    #데이터 형태 확인 <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))  #<class 'tuple'> 리스트와 똑같다는 뜻 그러나 리스트와 차이점은 튜플 소괄호 안에 생성되어 있고 한번 생성되면 바꿀 수가 없다.
# print(type(xy_train[0][0]))  #<class 'numpy.ndarray'> 
# print(type(xy_train[0][1]))  #<class 'numpy.ndarray'> 
