import numpy as np                 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets ['target']
print(x.shape, y.shape)    #(1797, 64) (1797,)
print(np.unique(y, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))


from tensorflow.keras.utils import to_categorical 
y = to_categorical(y) 

print(y)
print(y.shape)   #(1797, 10)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    shuffle=True, random_state=222, test_size=0.2, stratify=y)

scaler = MinMaxScaler()   #
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)   #minmaxscaler  
x_test = scaler.transform(x_test)


# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)     
# x_test = scaler.transform(x_test)



# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[3])
# plt.show()


#2.모델구성 
model= Sequential()
model.add(Dense(100, activation='relu', input_shape=(64, )))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(82, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(75, activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(10, activation='softmax')) 

#3.컴파일, 훈련

from tensorflow.keras.callbacks import EarlyStopping 

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])                 
earlystopping = EarlyStopping(monitor='val_loss', mode='min', 
                              patience=10, restore_best_weights=True, verbose=1)

model.fit(x_train, y_train, epochs=500, batch_size=1, 
          validation_split=0.2, callbacks=[earlystopping], verbose=1) 

#4. 평가 예측 
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


"""
    결과: accuracy :  0.9750000238418579
    
    minmaxscaler: accuracy :  0.09444444626569748
    standardscaler: accuracy :  0.10000000149011612
    
"""