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
    train_size=0.7, shuffle=True, random_state=123
) 
#
print(x)
print(x.shape) #(442, 10)
print(y)
print(y.shape) #(442,)
print ('x_train : ',x_train) 
print ('x_test : ', x_test)
print ('y_train : ', y_train) 
print ('y_test : ', y_test) 


print(dataset.feature_names)
print(dataset.DESCR)

#2.모델구성
model = Sequential()
model.add(Dense(19,input_dim=10))
model.add(Dense(20))
model.add(Dense(18))
model.add(Dense(80))
model.add(Dense(331))
model.add(Dense(7))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(2))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics= ['mae']) 

model.fit(x_train, y_train, epochs=1200, batch_size=15) 

#평가 예측 
loss = model.evaluate(x_test , y_test)
print('loss : ', loss)
# ㅣㅣ
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
print("R2 : ", r2)