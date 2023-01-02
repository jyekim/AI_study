import numpy as np 
import tensorflow as tf    # 코드가 시작하는 점을 위에다
from tensorflow.keras.models import Sequential #seqen 이라는 걸 당겨왔다
from tensorflow.keras.layers import Dense

#1.  데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

#2. 모델 구성
model = Sequential()  #이  seq 모델을 정의를 했고
model.add(Dense(3, input_dim=1)) # 리스트 하나씩 강의때 해준 그림 기억
model.add(Dense(50)) #첫번째 인풋 레이어를 뺀 나머지는 명시하지 않는다 
model.add(Dense(4))   # 훈련의 양을 조절할 수 있고 layer의 갯수 깊이 역시 조절이 가능하다 #하이퍼파라미터 튜닝  
model.add(Dense(20)) 
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mae', optimizer='adam')
model.fit(x, y, epochs=210)

#4.  평가
result = model.predict([6])
print('6의 결과:',result)


