import numpy as np         
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense    


#1.데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape)   #(2, 10) 10행2열이라는것             
print(y.shape)   #(10, )

x = x.T  # T는 행과 열을 바꾼다는 것 / t =  전치하는 것 
print(x.shape)  #(10,2)로 정제해줌

#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=2))  #두개가 들어가서 다섯개가 나옴 그림 생각 인풋딤은 행열에서 열의 개수와 같다/  열은 feature피처,컬럼이라고 한다 
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련(train set)
model.compile(loss='mae', optimizer='adam')
model.fit(x, y,epochs=150, batch_size=1) #훈련데이터의 과적합/ 훈련할때는 좋은 성능이 나오지만 예측할때는 엉망인 결과가 나올수도 있다.

#4 평가(test set) 예측
loss = model.evaluate(x, y)
print('loss :', loss)

result = model.predict([[10, 1.4]])
print('[10, 1.4]의 예측값 : ', result)

'''
결과:        
batch
loss : 0.06600888073444366
[10, 1.4]의 예측값 :  [[20.128046]]       

'''