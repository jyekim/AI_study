import numpy as np 
import tensorflow as tf    # 코드가 시작하는 점을 위에다
from tensorflow.keras.models import Sequential #seqen 이라는 걸 당겨왔다
from tensorflow.keras.layers import Dense

#1.  데이터
x = np.array([1,2,3,4,5,6]) 
y = np.array([1,2,3,5,4,6]) # batch는 전체데이터를 훈련 시키는것을 어떤 방식으로 할 것인지 선택할 수 있다. 예를 들면 배치단위를 잘게 자르고 한개씩 훈련을 시킬 경우 시간이 너무 오래 걸림  
                          
#2. 모델 구성
model = Sequential()  
model.add(Dense(3, input_dim=1)) 
model.add(Dense(50)) 
model.add(Dense(4))     
model.add(Dense(20)) 
model.add(Dense(1))

#3. 컴파일, 훈련              
model.compile(loss= 'mae', optimizer='adam')
model.fit(x, y, epochs=10, batch_size=2)  #배치는 fit 에서 조절을 한다 배치를 2로 줫을때 데이터는 2개씩 3번 작업함 / 딱떨어지지ㅣ 않는 나머지 배치 수 예를 들면 4도 4개 2개로 2배치가 된다. /전체 통배치 6으로 했을때는 그냥 1 번 통으로 함
#배치 사이즈가 7이면 데이터 보다 오버될 경우; 그냥 통채로 훈련 시킨다. 배치사이즈를 명시 하지 않았을때도 하이퍼파라미터의 디폴트값 있다.  디폴트값은 32 
#4.  평가
result = model.predict([6])
print('6의 결과:',result)

#블록처리해서 tab 누르면 줄 띄우기 됨   블록처리 후 다시 원위치 하고 싶을 때  shift+tab 

"""
@@@예시
이렇게 
할 수 있음 
이거는 블록 주석처리 되는 것

"""
#쌍따옴표 3개 쌍따옴표 1개 블록 주석처리 됨  