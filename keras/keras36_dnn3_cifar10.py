from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np
from tensorflow.keras.models import Sequential        
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = 'c:/study4/_save/'

#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)               
print(x_test.shape, y_test.shape)               #(50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))   #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000] dtype=int64))

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train/255.                            
x_test = x_test/255.


#2. 모델구성 
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(3072, )))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='linear'))                          
model.add(Dense(10, activation='softmax'))
model.summary()


#3.컴파일 훈련 
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
es= EarlyStopping(monitor='val_loss', mode='min', patience=15, 
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
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                      filepath= filepath + 'k36_03_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32,
          validation_split=0.2, callbacks=[es, mcp])


# model.save(path +"keras35_03_ModelCheckPoint1_save_model.hdf5")

#평가 예측   
results =model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])


"""

결과값 : 
Dnn Epoch 00052: early stopping
313/313 [==============================] - 1s 2ms/step - loss: 1.8729 - acc: 0.3263
loss :  1.8728822469711304
acc :  0.3262999951839447





"""