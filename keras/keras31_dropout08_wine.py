import numpy as np              
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


path = './_save/'


#1. 데이터   와인데이터 
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     #(178, 13) (178,)
print(y)
print(np.unique(y))         #[0 1 2]     #라벨의 유니크한 값을 찾는거임
print(np.unique(y, return_counts=True))        #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y) 
print(y.shape)  #(178, 3) 


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2, stratify=y)


# scaler = MinMaxScaler()   #
# scaler.fit(x_train)
# x_train = scaler.fit_transform(x_train)   #minmaxscaler  
# x_test = scaler.transform(x_test)


# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)     
# x_test = scaler.transform(x_test)

#2.모델구성 
# model= Sequential()
# model.add(Dense(100, activation='relu', input_shape=(13, )))
# model.add(Dropout(0.5))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(82, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(511, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(75, activation='linear'))
# model.add(Dense(9, activation='linear'))
# model.add(Dense(20, activation='linear'))
# model.add(Dense(3, activation='softmax')) 


# #2. 모델구성(함수형)
input1 = Input(shape=(13,))       #인풋레이어는 
dense1 = Dense(50, activation= 'relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(40, activation= 'sigmoid')(drop1)
drop2= Dropout(0.3)(dense2) 
dense3 = Dense(30, activation= 'relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(20, activation= 'linear')(drop3)
output1 = Dense(3, activation= 'softmax')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()



#3.컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es= EarlyStopping(monitor='val_loss', mode='min',
                              patience=10, restore_best_weights=True,
                              verbose=1) 

import datetime
date = datetime.datetime.now()
print(date)
print(type(date))
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date))

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbpse=1, save_best_only=True,
                      filepath= filepath +'k31_08_' + date + '_'+ filename)



hist = model.fit(x_train, y_train, epochs=500, batch_size=10,
          validation_split=0.2, callbacks=[es,mcp],
          verbose=1) 
model.save(path +"keras31_dropout08_save_model.hdf5")

#4.평가 예측 
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ,loss')  
print('accuracy : ', accuracy)                              

# print(y_test[:5])                                                       
# y_predict = model.predict(x_test[:5]) 
# print(y_predict)                                          

from sklearn.metrics import accuracy_score
import numpy as np                                                            

y_predict = model.predict(x_test)     
y_predict =np.argmax(y_predict, axis=1)   # 가장 큰 값을 찾아내는 것 # 가장 큰 자릿값 뽑아냄   / axis=1 (가로축(행)), axis=0 (세로축(열))
print('y_pred(예측값) :', y_predict)
y_test =np.argmax(y_test, axis=1)     
print('y_test(원래값) : ', y_test)  
print(y_predict)


acc = accuracy_score(y_test, y_predict)     # 소수점 들어가는 실수 형태로 구성// error 발생
print(acc)


"""결과
데이터가 적으니 스케일러 안하는게 더 나음
    accuracy :  0.9166666865348816
    
    스케일러 
 MinMaxScaler :   0.3888888955116272
 StandardScaler :   accuracy :  0.3888888955116272
    
    drop out : accuracy :  0.3888888955116272
"""