import numpy as np   
import tensorflow as tf    #tensorflow 가져올거임 그런데 이거를 줄여서 tf로 하는거   
print(tf.__version__)    # 2.7.4

#1. (정제된) 데이터준비 
x= np.array([1,2,3,4,5])
y= np.array([1,2,3,5,4])

#2. 모델구성 y=wx+b 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential ()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam' )
model.fit(x, y, epochs=220)

#4. 평가, 예측
results = model.predict([6])
print('6의 예측값:',results)
