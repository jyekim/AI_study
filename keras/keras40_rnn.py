import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터

dataset = np.array([1,2,3,4,5,6,7,8,9,10])   #데이터의 형태는 (10, )
#실질적으로 y는 없음
x = np.array ([[1,2,3], 
               [2,3,4], 
               [3,4,5], 
               [4,5,6],
               [5,6,7], 
               [6,7,8], 
               [7,8,9]])

y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape)  #(7, 3) (7,)

x = x.reshape(7, 3, 1)
print(x.shape)           #(7, 3, 1)  ===> [[[1],[2],[3]], [2],[3],[4]],.....]

#2. 모델구성

model = Sequential()
model.add(SimpleRNN(64, input_shape=(3, 1), activation='relu'))          #이부분이 rnn이 되는 것 
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='linear'))
model.add(Dense(10, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=2, verbose=1)

#4. 평가예측

loss = model.evaluate(x, y)
print(loss) 
y_pred = np.array([8, 9, 10]).reshape(1, 3, 1)    #스칼라3개짜리 데이터하나를 백터 1로 바꿈 
result = model.predict(y_pred)
print('[8, 9, 10]의 결과 : ', result)