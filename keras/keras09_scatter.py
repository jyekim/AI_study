from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np                   
from sklearn.model_selection import train_test_split
#scatter 산점도표현하기 
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
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1) 

#평가 예측 
loss = model.evaluate(x_test ,y_test)
print('loss : ', loss)
y_predict = model.predict(x)
import matplotlib.pyplot as plt
plt.scatter(x, y) 
plt.plot(x, y_predict, color='red') #예상값을 선으로 쭉이어지는것 
plt.show()   
