from tensorflow.keras.datasets import fashion_mnist
import numpy as np 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)               #(60000, 28, 28) (60000,)   뒤에 1이 없으니 흑백데이터 라는 것을 인지 
print(x_test.shape, y_test.shape)               #(10000, 28, 28) (10000,)

print(x_train[1000])
print(y_train[1000])
 
path = 'c:/study4/_save/'
# import matplotlib.pyplot as plt
# plt.imshow(x_train[1000],'gray')
# plt.show()

#1. 데이터 
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)      #(60000, 28, 28, 1) (60000,)
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

print(x_train.shape, y_train.shape) 
print(x_test, y_test.shape)


#2. 모델구성 
model = Sequential()
input1 = Input(shape=(28*28*1))       #인풋레이어는 
dense1 = Dense(50, activation= 'relu')(input1)
dense2 = Dense(40, activation= 'sigmoid')(dense1)
dense3 = Dense(30, activation= 'relu')(dense2)
dense4 = Dense(20, activation= 'linear')(dense3)
output1 = Dense(1, activation= 'linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()

#3.컴파일 훈련 
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
es= EarlyStopping(monitor='val_loss', mode='min', patience=10, 
                  restore_best_weights=True, verbose=1)



import datetime
date = datetime.datetime.now()                #현재 시간이 저장된다 
print(date)                                  #2023-01-12 14:58:05.884579
print(type(date))                            #<class 'datetime.datetime'>   
date = date.strftime("%m%d_%H%M")             #0112_1503   오늘의 날짜와 시간                    
print(date)
print(type(date))
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, ave_best_only=True,
                      filepath= filepath + 'k38_02_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=50, verbose=1, batch_size=32,
          validation_split=0.2, callbacks=[es, mcp])

# model.save(path +"keras38_2_ModelCheckPoint1_save_model.hdf5")

#평가 예측   
results =model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])


"""결과
Epoch 00019: early stopping
313/313 [==============================] - 1s 4ms/step - loss: 0.3188 - acc: 0.8948
loss :  0.31878843903541565
acc :  0.8948000073432922
    """
