#실습 R2 0.62이상

from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout 
import numpy as np         
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터

dataset = load_diabetes()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test= train_test_split(x, y,
    train_size=0.7, random_state=66
) 

# print(x)
# print(x.shape) #(442, 10)
# print(y)
# print(y.shape) #(442,)
# print ('x_train : ',x_train) 
# print ('x_test : ', x_test)
# print ('y_train : ', y_train) 
# print ('y_test : ', y_test) 

scaler = MinMaxScaler()   #
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)   #minmaxscaler  
x_test = scaler.transform(x_test)


# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)     
# x_test = scaler.transform(x_test)



# #2.모델구성(순차형)
# model = Sequential()
# model.add(Dense(19, input_dim=10))
# model.add(Dropout(0.5))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(18, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(78, activation='relu'))
# model.add(Dense(18, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(1, activation='linear'))

# #2. 모델구성(함수형)
input1 = Input(shape=(10,))       #인풋레이어는 
dense1 = Dense(50, activation= 'relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(40, activation= 'sigmoid')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(30, activation= 'relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(20, activation= 'linear')(drop3)
output1 = Dense(1, activation= 'linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam') 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                             patience=10, restore_best_weights=True, verbose=1)


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
                      filepath= filepath + 'k31_03' + date + '_' + filename)

hist = model.fit(x_train, y_train, epochs=200, batch_size=5, callbacks=[es,mcp], validation_split=0.2)





loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
#verbose 1  걸린시간
print("=================================")
print(hist)  #<keras.callbacks.History object at 0x000001762F26B5B0>\
print("==================================")
print(hist.history)      # dictionary 키 value 형태로 되어 있다. list형태 2개이상 /반환값 안에는  loss와 valloss의 dictionary히스토리에 제공된 변수가 있다는 뜻 
print("==================================")
print(hist.history['loss'])      
print("==================================")
print(hist.history['val_loss'])     

# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))     #.리스트 형태로 순서대로 되어 있는 것은  x를 명시 안해도 상관없다. 즉, y 만 넣어주면 됨
# plt.plot(hist.history['loss'], c= 'red', marker='.', label='loss')
# plt.plot(hist.history['val_loss'], c= 'blue', marker='.', label='val_loss')
# plt.grid() #격자
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('diabete loss')
# plt.legend()   # 라벨이 명시됨
# #plt.legend(loc='upper left')   
# plt.show()



#4평가 예측 
loss = model.evaluate(x_test , y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
# print("===================")
# print(y_test)
# print(y_predict)
# print("====================")
#modelpredict에 최적의 가중치가 생성되어 있다/ x_test예측값이 생성되어서 y_predict에 넣어놓겠다 


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict)) # RMSE ytest와 ypredict값을 받아서 
print("RMSE : ", RMSE(y_test, y_predict))  

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


"""
결과 
loss :  3155.137451171875
RMSE :  56.17061103226517
R2 :  0.4935874757693879


MinMaxScaler : loss :  7466.2568359375
               RMSE :  86.40750642761313
               R2 :  -0.1983649162993668
               
               
StandardScaler : loss :  7908.8291015625
RMSE :  88.93159879425869
R2 :  -0.2693995541627887             

"""