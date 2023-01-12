from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten      #conv2D 이차원 이미지쪽은 이렇게 함 

model = Sequential()
                                        #(2, 2)는 조각을 내는것
model.add(Conv2D(filters=10, kernel_size=(2,2),                      
                 input_shape=(5, 5, 1)))
model.add(Conv2D(filters=5, kernel_size=(2,2)))        #dense모델을 할때 히든레이어는 사람이 정하듯 output필터 역시 정하면됨
                                                      #                      
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))

model.summary()