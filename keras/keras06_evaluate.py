import numpy as np 
import tensorflow as tf    # 코드가 시작하는 점을 위에다
from tensorflow.keras.models import Sequential #seqen 이라는 걸 당겨왔다
from tensorflow.keras.layers import Dense

#1.  데이터
x = np.array([1,2,3,4,5,6]) 
y = np.array([1,2,3,5,4,6])  
                          
#2. 모델 구성
model = Sequential()  
model.add(Dense(10, input_dim=1)) 
model.add(Dense(10)) 
model.add(Dense(10))   
model.add(Dense(10)) 
model.add(Dense(1))

#3. 컴파일, 훈련              
model.compile(loss= 'mae', optimizer='adam')
model.fit(x, y, epochs=10, batch_size=2)  #배치는 fit 에서 조절을 한다 배치를 2로 줫을때 데이터는 2개씩 3번 작업함 / 딱떨어지지ㅣ 않는 나머지 배치 수 예를 들면 4도 4개 2개로 2배치가 된다. /전체 통배치 6으로 했을때는 그냥 1 번 통으로 함
#배치 사이즈가 7이면 데이터 보다 오버될 경우; 그냥 통채로 훈련 시킨다. 배치사이즈를 명시 하지 않았을때도 하이퍼파라미터의 디폴트값 있다. 

#4.  평가,예측
loss = model.evaluate(x, y)   #.평가한다는 건 loss가 낮을 수록 좋다는 것이기에 x,y를 넣었을 때 어떻게 나오냐면 loss값이 나온담. evaluate가 들어가는 평가데이터는 나중에 model 데이커가 들어가면 안됨
print('loss: ', loss)      
result = model.predict([6])
print('6의 결과:',result)

#판단의 기준은  항상 loss가 기준이다. 다른 지표도 나오긴 하지만...그래도 loss가 가장 중요 
