#43 카피함

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
model = Sequential()
model.add(GRU(units=13, input_shape=(3,1)))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1))
model.summary()


#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping    #대문자로 시작하면 파이썬의 클래스로 지정되어 있는것/ 함수는 소문자로 시작함
earlystopping = EarlyStopping(monitor='val_loss', mode='min', 
                              patience=10, restore_best_weights=True, verbose=1)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=150, batch_size=1, verbose=1, callbacks=[earlystopping])

#4. 평가예측

loss = model.evaluate(x, y)
print(loss) 
y_pred = np.array([50,60,70]).reshape(1, 3, 1)    
result = model.predict(y_pred)
print('[50,60,70]의 결과 : ', result)
