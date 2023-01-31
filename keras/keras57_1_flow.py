import numpy as np  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
augument_size =100


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
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1), # x     #reshape에 -1은 전체데이터를 뜻함 
    np.zeros(augument_size),                                                  # y 
    batch_size=augument_size, 
    shuffle=True
)

print(x_data[0])
print(x_data[0][0].shape)  #(100, 28, 28, 1)
print(x_data[0][1].shape)  #(100,)
 
import matplotlib.pyplot as plt 
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')

plt.show()



