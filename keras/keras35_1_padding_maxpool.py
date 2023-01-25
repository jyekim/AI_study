import numpy as np           
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)               #(60000, 28, 28) (60000,)   뒤에 1이 없으니 흑백데이터 라는 것을 인지 
print(x_test.shape, y_test.shape)               #(10000, 28, 28) (10000,)

path = 'c:/study4/_save/'

#1. 데이터
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, y_train.shape)               #(60000, 28, 28, 1) (60000,)
                                                  #(10000, 28, 28, 1) (10000,)
# print(x_test.shape, y_test.shape)      

# print(np.unique(y_train, return_counts=True))


#2. 모델구성 
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu', padding='same', strides=1))  #(28,28,128) #패딩이 valid와 same일때 값 다른거 찍어내기  #stride로 하면 2로 나눠짐 
#model.add(MaxPooling2D())                          #(14, 14, 128)
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same'))     #(28, 28, 128)
model.add(MaxPooling2D()) 
model.add(Conv2D(filters=64, kernel_size=(2,2)))     #(25, 25, 64)
model.add(Flatten())                                 # 400000
model.add(Dense(32, activation='relu'))              #input_shape= (600000, 400000) 근데 행 무시하니깐 결과적으로 (400000, )
                #(6만, 4만)이 인풋 = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))

model.summary() #padding을 적용하는 경우 늘어나는 것을 볼 수 있음 찍어내서 보기  

"""#3.컴파일 훈련 
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
                      filepath= filepath + 'k35_01_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32,
          validation_split=0.2, callbacks=[es, mcp])

model.save(path +"keras35_ModelCheckPoint1_save_model.hdf5")

#평가 예측   
results =model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

"""
# es/ mcp  validation 적용  


""" 

기존성능
결과 Epoch 00014: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1183 - acc: 0.9689
loss :  0.1182783842086792
acc :  0.9689000248908997    
    
 패딩 적용시
 결과 Epoch 00018: early stopping
313/313 [==============================] - 1s 4ms/step - loss: 0.1248 - acc: 0.9672
loss :  0.12480848282575607
acc :  0.967199981212616
    
    
  maxpooling 적용시
  결과 Epoch 00016: early stopping
loss :  0.08505487442016602
acc :  0.9782000184059143


  maxpooling 2번적용시
  Epoch 00019: early stopping
313/313 [==============================] - 1s 4ms/step - loss: 0.0719 - acc: 0.9838
loss :  0.07192462682723999
acc :  0.9837999939918518
    """