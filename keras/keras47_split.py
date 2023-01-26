#시계열 데이터를 할때 이 데이터를 어떻게 자를 것인지 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

a = np.array(range (1, 11))
timesteps = 5

def split_x(dataset, timesteps):
    aaa =[]                           #빅리스트 만든것 
    for i in range(len(dataset) - timesteps + 1):   #5-3+1 만큼 반복 
        subset = dataset[i : ( i + timesteps)]          # a[0:(0+3)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape)   #(6,5)

x = bbb[:, :-1]   #모든 행의 하나 빼는것 
y = bbb[:, -1]
print(x , y) 
print(x.shape, y.shape)  #(6, 4) (6,)

x = x.reshape(6, 4, 1)
print(x.shape) 

# 실습 
# LSTM  모델 구성

x_predict = np.array([7, 8, 9, 10])


#2. 모델 구성
model = Sequential()
model.add(LSTM(units=6, input_shape=(4,1), activation='relu'))  
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, verbose=1)

#4. 평가예측

loss = model.evaluate(x, y)
print(loss) 
y_pred = np.array([7, 8, 9, 10]).reshape(1, 4, 1)    
result = model.predict(y_pred)
print('[7, 8, 9, 10]의 결과 : ', result)


"""[7, 8, 9, 10]의 결과 :  [[10.60672]]
 activations을 넣었을 때 [7, 8, 9, 10]의 결과 :   [[11.019175]]
"""