from sklearn.datasets import load_iris   
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터

datasets = load_iris()
# print(datasets.DESCR)      # pandas 에서는 .describe() 혹은 .info()
# print(datasets.feature_names)   #pandas .columns 으로 씀 괄호가 


x = datasets.data
y = datasets['target']
# print(x) 
# print(y)
print(x.shape)
print(y.shape)   #(150, 4),  (150, )


from tensorflow.keras.utils import to_categorical 
y = to_categorical(y) 

print(y)
print(y.shape) # (150, 3) 으로 바뀐걸 알 수 있음 



x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True,
                                                     random_state=333, test_size=0.2, stratify=y)   #false의 문제점은?  stratify=ㅛ 옵션을 넣어주면 한쪽으로 치우쳐지는거 배제됨


# scaler = MinMaxScaler()   #
# scaler.fit(x_train)
# x_train = scaler.fit_transform(x_train)   #minmaxscaler  
# x_test = scaler.transform(x_test)


# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)     
# x_test = scaler.transform(x_test)
print(x)
print(type(x)) 


# print(y_train)
# print(y_test)

# #2. 모델구성
# model= Sequential()
# model.add(Dense(100, activation='relu', input_shape=(4, )))
# model.add(Dense(40, activation='sigmoid'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='linear'))
# model.add(Dense(3, activation='softmax'))           #다중분류는 마지막레이어 activation=softmax로 해야함
                                                    #3인 이유는 y 안에 들어가는 class가 3개이기 때문 마지막 노드의 갯수는 클래스의 갯수와 동일하게 해준다

# #2. 모델구성(함수형)
input1 = Input(shape=(4,))       #인풋레이어는 
dense1 = Dense(50, activation= 'relu')(input1)
dense2 = Dense(40, activation= 'sigmoid')(dense1)
dense3 = Dense(30, activation= 'relu')(dense2)
dense4 = Dense(20, activation= 'linear')(dense3)
output1 = Dense(3, activation= 'softmax')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()




#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])           

      
model.fit(x_train, y_train, epochs=500, batch_size=1,
          validation_split=0.2,
          verbose=1) 


#평가 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ,loss')  
print('accuracy : ', accuracy)                              

# print(y_test[:5])                                                       
# y_predict = model.predict(x_test[:5]) 
# print(y_predict)                                          

from sklearn.metrics import accuracy_score
import numpy as np                                                            

y_predict = model.predict(x_test)     
y_predict =np.argmax(y_predict, axis=1)   
print('y_pred(예측값) :', y_predict)
y_test =np.argmax(y_test, axis=1)     # 가장 큰 값을 찾아내는 것 
print('y_test(원래값) : ', y_test)  
print(y_predict)


acc = accuracy_score(y_test, y_predict)
print(acc)



"""결과 
데이터가 적으니 스케일러 안하는게 더 나음
loss : ,loss
accuracy :  0.8999999761581421
"""