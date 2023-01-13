from tensorflow.keras.datasets import cifar10, cifar100
import numpy as np
from tensorflow.keras.models import Sequential        
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = 'c:/study4/_save/'

#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape)               
# print(x_test.shape, y_test.shape)               #(50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train, return_counts=True))   #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000] dtype=int64))



#2. 모델구성 
model = Sequential()
model.add(Conv2D(filters=200, kernel_size=(3,3), input_shape=(32, 32, 3), activation='relu')) #(31, 31, 200)
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=100, kernel_size=(2,2)))#(30, 30, 100)
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=70, kernel_size=(3,3), activation='relu')) 
model.add(Flatten())                                 
model.add(Dense(10, activation='softmax'))
# model.summary()


#3.컴파일 훈련 
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
es= EarlyStopping(monitor='val_loss', mode='min', patience=20, 
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
                      filepath= filepath + 'k34_02_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32,
          validation_split=0.2, callbacks=[es, mcp])

# model.save(path +"keras34_02_ModelCheckPoint1_save_model.hdf5")

#평가 예측   
results =model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])



"""

결과값 : 


model = Sequential()
model.add(Conv2D(filters=200, kernel_size=(3,3), input_shape=(32, 32, 3), activation='relu')) #(31, 31, 128)
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=100, kernel_size=(2,2)))#(30, 30, 64)
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=70, kernel_size=(3,3), activation='relu')) 
model.add(Flatten())                                 # 53824
model.add(Dense(10, activation='softmax'))
Epoch 00015: val_loss improved from 1.11781 to 1.06644, saving model to ./_save/MCP\k34_02_0113_1919_0015-1.0664.hdf5
178/178 [==============================] - 3s 15ms/step - loss: 0.8310 - acc: 0.7124 - val_loss: 1.0664 - val_acc: 0.6441







    """