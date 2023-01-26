#45-1 카피함
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터

x = np.array ([[1,2,3], [2,3,4], [3,4,5], 
               [4,5,6],[5,6,7], [6,7,8], 
               [7,8,9],[8,9,10],[9,10,11],
               [10,11,12],[20,30,40],[30,40,50],
               [40,50,60]])

y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60,70])
y_predict = np.array([50,60,70])    #80 만들기
print(x.shape, y.shape)   #(13, 3) (13,)

x = x.reshape(13, 3, 1)
print(x.shape) 


#2. 모델구성 
model = Sequential()               #(N, 3, 1)
model.add(LSTM(units=64, input_shape=(3,1), return_sequences=True))     #(N, 64)    / return_sequences를 써주면 LSTM을 두번 써줄 수 있다  
model.add(LSTM(32))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(1))
model.summary()


"""#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping    #대문자로 시작하면 파이썬의 클래스로 지정되어 있는것/ 함수는 소문자로 시작함
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=150, batch_size=5, verbose=1)

#4. 평가예측

loss = model.evaluate(x, y)
print(loss) 
y_pred = np.array([50,60,70]).reshape(1, 3, 1)    
result = model.predict(y_pred)
print('[50,60,70]의 결과 : ', result)
"""

"""
[50,60,70]의 결과 :  [[74.47937]]
    """