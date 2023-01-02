import tensorflow as tf
print(tf.__version__)
import numpy as np

#1.   데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2.  모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1)) #dense레이어 안에 넣겟다 와이가 1, inputdim이 x  

#3. 컴파일과 훈련 
model.compile(loss='mae', optimizer='adam') #mean average error ?
model.fit(x, y, epochs=2000) #fit은훈련을 시켜라 뜻,epochs는 훈련을 몇번 할건인지 /컨트롤에프오 하면 로스가 줄어듬
#훈련을 너무 많이 하면 성능이 좋아지다가도 안될 수가 있음 수치를 적당히 찾아야함 

#4. 평가, 예측
result = model.predict([4]) #예측할 값들
print('결과 : ', result)










