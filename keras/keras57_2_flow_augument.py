import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
augument_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augument_size)  #6만개 중에 4만개를 랜덤으로 뽑는 것

print(randidx)        #  [14072 15943   563 ... 19535 49990 18681]
print(len(randidx))   #  40000

x_augument = x_train[randidx].copy()       # 데이터 원본 엑스트레인을 건들지 않고 복사본을 augument에 넣음
y_augument = y_train[randidx].copy()
print(x_augument.shape, y_augument.shape)  #(40000, 28, 28) (40000,)

x_augument = x_augument.reshape(40000, 28, 28, 1)


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
x_augumented = train_datagen.flow(
    x_augument,
    y_augument,                                               
    batch_size=augument_size, 
    shuffle=True
)

print(x_augumented[0][0].shape)  #(40000, 28, 28, 1)
print(x_augumented[0][1].shape)  #(40000,)

x_train = x_train.reshape(60000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented[0][0]))
y_train = np.concatenate((y_train, x_augumented[0][1]))

print(x_train.shape, y_train.shape)  #(100000, 28, 28, 1) (100000,)