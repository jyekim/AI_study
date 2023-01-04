from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np                   
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test= train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123
) 

#2. 모델구성 
model = Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(10))
model.add(Dense(1035))
model.add(Dense(7))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics= ['mae']) 

model.fit(x_train, y_train, epochs=200, batch_size=1) 

#평가 예측 
loss = model.evaluate(x_test , y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print("===================")
print(y_test)
print(y_predict)
print("====================")
#modelpredict에 최적의 가중치가 생성되어 있다/ x_test예측값이 생성되어서 y_predict에 넣어놓겠다 

from sklearn.metrics import mean_squared_error 
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict)) # RMSE ytest와 ypredict값을 받아서 


print("RMSE : ", RMSE(y_test, y_predict))  
    
#def는 함수를 정의하는 것
#처음 훈련할때만 여러번 훈련시키고 그 다음 좋은 가중치만 빼면 됨
# RMSE :  3.929725415854297
# RMSE :  4.079660468256004