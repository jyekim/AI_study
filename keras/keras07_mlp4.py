import numpy as np                      
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 
x = np.array(range(10))  # (10, ) (10, 1) 은 동일한 뜻
print(x.shape)  
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])
print(y.shape) 

y = y.T  
print(y.shape)

#2.모델구현
model = Sequential()
model.add(Dense(3,input_dim=1))  
model.add(Dense(50))
model.add(Dense(420))
model.add(Dense(164))
model.add(Dense(5))
model.add(Dense(2640))
model.add(Dense(90))
model.add(Dense(7))
model.add(Dense(900))
model.add(Dense(3))

#3.컴파일
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=2)

#4.평가
loss = model.evaluate(x, y)
print('loss:',loss)
result = model.predict([[9]])  #행의 개수는 달라도 열 우선 / 특성값은 동일하게 줌
print('[9]의 예측값 : ', result)


""" 결과

model.add(Dense(3,input_dim=1))  
model.add(Dense(50))
model.add(Dense(83))
model.add(Dense(10))
model.add(Dense(71))
model.add(Dense(7))
model.add(Dense(22))
model.add(Dense(3))

batch_size=2 
epochs=100 
loss: 0.20636682212352753
[9]의 예측값 :  [[10.152039   1.3959985  0.6018365]] 


model.add(Dense(3,input_dim=1))  
model.add(Dense(50))
model.add(Dense(83))
model.add(Dense(164))
model.add(Dense(71))
model.add(Dense(7))
model.add(Dense(900))
model.add(Dense(3))
epochs=100, 
batch_size=2
loss: 0.1412142813205719
[9]의 예측값 :  [[10.13014     1.7055348   0.14067863]]

model.add(Dense(3,input_dim=1))  
model.add(Dense(50))
model.add(Dense(420))
model.add(Dense(164))
model.add(Dense(5))
model.add(Dense(2640))
model.add(Dense(90))
model.add(Dense(7))
model.add(Dense(900))
model.add(Dense(3))
loss: 0.5217419862747192
[9]의 예측값 :  [[10.384074    1.4742362  -0.55794007]]

"""