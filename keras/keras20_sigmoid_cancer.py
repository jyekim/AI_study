from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np 

#1. 데이터

datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)  #['mean radius' 'mean texture' 'mean perimeter' 'mean ... 30개의 컬럼이 있음
x = datasets['data']
y = datasets['target']
# print(x.shape, y.shape)'
x_train , x_test, y_train , y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2)

#2, 모델구성
model = Sequential()
model.add(Dense(30, activation= 'linear', input_shape=(30, )))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))  
 
 
 
#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])       

from tensorflow.keras.callbacks import EarlyStopping    
# earlystopping = EarlyStopping(monitor='accuray', mode='auto', 
earlystopping = EarlyStopping(monitor='val_loss', mode='min', 
                              patience=20, restore_best_weights=True, verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=15, validation_split=0.2, callbacks=[earlystopping], verbose=1)

#4, 평가 예측 
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
y_predict_2 = np.where(y_predict >= 0.5, 1, 0 ) #과제1 실수값을 정수형으로 바꿔주면 됨 R로 시작되는어떤거  #과제2  accuracy score 완성시키기 

# 방법 2        intarr = list(map(int, y_predict))
# 방법 3        y_predict = np.asarray(y_predict, dtype = int)    # np.asarray: 입력된 데이터를 np.array 형식으로 만듬. #(import numpy as np로 임포트 안했으면 np 대신에 numpy 그대로 입력해야함.) 
                                                                    # dtype 속성: 데이터 형식 변경 
# (int: 정수형 / float: 실수형 / complex: 복소수형 / str: 문자형)
print(y_predict[:10])
print(y_predict_2[:10])

# print("==============")
# print(y_predict_2)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict_2)
print("accuracy_score : ", acc)