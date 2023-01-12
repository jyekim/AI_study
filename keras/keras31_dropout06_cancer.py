from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터

path = './_save/'

datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)  #['mean radius' 'mean texture' 'mean perimeter' 'mean ... 30개의 컬럼이 있음
x = datasets['data']
y = datasets['target']
# print(x.shape, y.shape)'
x_train , x_test, y_train , y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2)



scaler = MinMaxScaler()   #
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)   #minmaxscaler  
x_test = scaler.transform(x_test)


# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)     
# x_test = scaler.transform(x_test)


#2, 모델구성
# model = Sequential()
# model.add(Dense(30, activation= 'linear', input_shape=(30, )))
# model.add(Dropout(0.5))
# model.add(Dense(40, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(30, activation= 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(20, activation= 'relu'))
# model.add(Dense(1, activation= 'sigmoid'))  
 

# #2. 모델구성(함수형)
input1 = Input(shape=(30,))       #인풋레이어는 
dense1 = Dense(50, activation= 'relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(40, activation= 'sigmoid')(drop1)
drop2= Dropout(0.3)(dense2) 
dense3 = Dense(30, activation= 'relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(20, activation= 'linear')(drop3)
output1 = Dense(1, activation= 'linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()


 
#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])       

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint   
# earlystopping = EarlyStopping(monitor='accuray', mode='auto', 
es = EarlyStopping(monitor='val_loss', mode='min', 
                              patience=10, restore_best_weights=True, verbose=1)

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
                      filepath= filepath +'k31_06_' + date + '_'+ filename)


model.fit(x_train, y_train, epochs=10000, batch_size=15, validation_split=0.2, callbacks=[es, mcp], verbose=1)

model.save(path +"keras31_dropout06_save_model.hdf5")

#4, 평가 예측 
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
y_predict_2 = np.where(y_predict >= 0.5, 1, 0 ) #과제1 실수값을 정수형으로 바꿔주면 됨 R로 시작되는어떤거  #과제2  accuracy score 완성시키기 

# 방법 2        intarr = list(map(int, y_predict))
# 방법 3        y_predict = np.asarray(y_predict, dtype = int)    # np.asarray: 입력된 데이터를 np.array 형식으로 만듬. #(import numpy as np로 임포트 안했으면 np 대신에 numpy 그대로 입력해야함.) 
                                                                    # dtype 속성: 데이터 형식 변경 
# (int: 정수형 / float: 실수형 / complex: 복소수형 / str: 문자형)
print(y_predict[:10])
print(y_predict_2[:10])

# print("==============")
# print(y_predict_2)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict_2)
print("accuracy_score : ", acc)


"""결과


dropout 후 : accuracy_score :  0.9298245614035088


    MinMaxScaler loss :  0.6723323464393616
                accuracy :  0.6754385828971863
    
    StandardScaler
    
    
    
"""