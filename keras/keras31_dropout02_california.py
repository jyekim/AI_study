#실습
#R2 0.55~0.6 


from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
import numpy as np         
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler


path = './_save/'


#1. 데이터  캘리 
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test= train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123
) 
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
# model.add(Dense(10, input_dim=8))
# model.add(Dropout(0.5))
# model.add(Dense(45, activation='relu'))
# model.add(Dense(15, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(53, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(22, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(90, activation='relu'))
# model.add(Dense(52, activation='relu'))
# model.add(Dense(83, activation='relu'))
# model.add(Dense(1, activation='linear'))




#2. 모델구성(함수형)
input1 = Input(shape=(8,))       #인풋레이어는 
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







#3.컴파일 훈련

model.compile(loss='mae', optimizer='adam') 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
                      filepath= filepath +'k31_02_' + date + '_'+ filename)

hist = model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2, callbacks=[es,mcp], verbose=1) 

model.save(path +"keras31_dropout02_save_model.hdf5")

#4. 평가 예측
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
# plt.title('california loss')
# plt.legend()   # 라벨이 명시됨
# #plt.legend(loc='upper left')   
# plt.show()


#평가 예측 
loss = model.evaluate(x_test , y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print("===================")
print(y_test)
print(y_predict)
print("====================")
#modelpredict에 최적의 가중치가 생성되어 있다/ x_test예측값이 생성되어서 y_predict에 넣어놓겠다 

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict)) # RMSE ytest와 ypredict값을 받아서 
print("RMSE : ", RMSE(y_test, y_predict))  

r2 = r2_score(y_test, y_predict)
print("R2스코어: ", r2)
#r2는높으면 성능이 좋다라는 것을 알 수 있다. 


"""

결과

RMSE :  0.7219690915215492
R2 :  0.6058052861157378

스케일링 후 : 
MinMaxScaler(R2 :  

StandardScaler : RMSE :  0.69273476311703
                  R2 :  0.6370828013168781


    """