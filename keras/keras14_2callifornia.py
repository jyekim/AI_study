#실습
#R2 0.55~0.6 


from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import numpy as np         
from sklearn.model_selection import train_test_split 


#1. 데이터 보스턴 집값에 대한 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test= train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123
) 
print ('x_train : ',x_train) 
print ('x_test : ', x_test)
print ('y_train : ', y_train) 
print ('y_test : ', y_test) 

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(x.shape) #(20640, 8)
print(y)
print(y.shape) #(20640,)


print(dataset.feature_names)
print(dataset.DESCR)

#2.모델구성
model = Sequential()
model.add(Dense(10,input_dim=8))
model.add(Dense(245))
model.add(Dense(15))
model.add(Dense(12))
model.add(Dense(53))
model.add(Dense(22))
model.add(Dense(12))
model.add(Dense(32))
model.add(Dense(90))
model.add(Dense(6))
model.add(Dense(83))
model.add(Dense(52))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics= ['mae']) 

model.fit(x_train, y_train, epochs=750, batch_size=20) 

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
print("R2 : ", r2)
#r2는높으면 성능이 좋다라는 것을 알 수 있다. 

"""
결과
model = Sequential()
model.add(Dense(10,input_dim=8))
model.add(Dense(245))
model.add(Dense(15))
model.add(Dense(12))
model.add(Dense(33))
model.add(Dense(22))
model.add(Dense(84))
model.add(Dense(52))
model.add(Dense(1))
epochs=150, batch_size=100) 
RMSE :  0.8335718761443435
R2 :  0.47451576324972466


model = Sequential()
model.add(Dense(10,input_dim=8))
model.add(Dense(245))
model.add(Dense(15))
model.add(Dense(12))
model.add(Dense(33))
model.add(Dense(22))
model.add(Dense(84))
model.add(Dense(52))
model.add(Dense(1))
epochs=300, batch_size=50
RMSE :  0.8063874511775855
R2 :  0.5082310465813582

model = Sequential()
model.add(Dense(10,input_dim=8))
model.add(Dense(245))
model.add(Dense(15))
model.add(Dense(12))
model.add(Dense(33))
model.add(Dense(22))
model.add(Dense(84))
model.add(Dense(52))
model.add(Dense(1))
epochs=600, batch_size=30) 
RMSE :  0.7712979223177887
R2 :  0.5500980156921897

RMSE :  0.7715834528323051
R2 :  0.5497648512453621
    
    """