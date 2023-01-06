#실습 R2 0.62이상

from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import numpy as np         
from sklearn.model_selection import train_test_split 


#1. 데이터

dataset = load_diabetes()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test= train_test_split(x, y,
    test_size=0.2, random_state=123
) 

# print(x)
# print(x.shape) #(442, 10)
# print(y)
# print(y.shape) #(442,)
# print ('x_train : ',x_train) 
# print ('x_test : ', x_test)
# print ('y_train : ', y_train) 
# print ('y_test : ', y_test) 



#2.모델구성
model = Sequential()
model.add(Dense(19, input_dim=10, activation = 'relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(81, activation='relu'))
model.add(Dense(66, activation='relu'))
model.add(Dense(58, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

#3.컴파일 훈련
import time
start = time.time()
model.compile(loss='mse', optimizer='adam') 

model.fit(x_train, y_train, epochs=500, batch_size=20, validation_split=0.3)
end = time.time()
print("걸린시간 : ", end - start)


#4평가 예측 
loss = model.evaluate(x_test , y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
# print("===================")
# print(y_test)
# print(y_predict)
# print("====================")
#modelpredict에 최적의 가중치가 생성되어 있다/ x_test예측값이 생성되어서 y_predict에 넣어놓겠다 

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict)) # RMSE ytest와 ypredict값을 받아서 
print("RMSE : ", RMSE(y_test, y_predict))  

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


"""
결과 
#2.모델구성
model = Sequential()
model.add(Dense(19, input_dim=10, activation = 'relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(81, activation='relu'))
model.add(Dense(66, activation='relu'))
model.add(Dense(58, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

#3.컴파일 훈련
import time
start = time.time()
model.compile(loss='mse', optimizer='adam') 

model.fit(x_train, y_train, epochs=500, batch_size=20, validation_split=0.3)

loss :  5018.89111328125
RMSE :  70.84413018956936
R2 :  0.2033718374145761
"""
