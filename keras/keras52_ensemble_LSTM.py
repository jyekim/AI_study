import numpy as np                      
import pandas as pd              
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input
# #numpy 데이터 불러오기 
# samsung = np.load('./_data/samsung.npy')
# amore = np.load('./_data/amore.npy')

# print(samsung)
# print(amore)
# print(samsung.shape)
# print(amore.shape)


#1. 데이터 
path = './_data/samsung/'
df1 = pd.read_csv(path + 'amore.csv',
                   encoding='cp949', sep=',',thousands=',')
print(df1)
print(df1.shape)    #(1980, 16)

df2 = pd.read_csv(path + 'samsung.csv',
                   encoding='cp949', sep=',', thousands=',')
print(df2)
print(df2.shape)   #(2221, 16)
print(df1.dtypes)

#일자 오름차순(최근날짜를 가장 아래로)
df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True]) 

df1['일자'] = pd.to_datetime(df1['일자'])
df1 = df1[df1['일자'] > '2018-05-03']
df2['일자'] = pd.to_datetime(df2['일자'])
df2 = df2[df2['일자'] > '2018-05-03']
print(df1)
print(df2)

df1 = df1[['시가','종가','저가','고가','거래량']]
df2 = df2[['시가','종가','저가','고가','거래량']]

df1.isnull().any()    #null값이 어느 열에 있는지
df2.isnull().any()

# # 삼성전자의 모든 데이터
# for i in range(len(df1.index)):       # 거래량 str 을 int 변경
#          for j in range(len(df1.iloc[i])):
#                 df1.iloc[i,j] = int(df1.iloc[i,j].replace(',', ''))
# # 아모레의 모든 데이터
# for i in range(len(df2.index)):
#          for j in range(len(df2.iloc[i])):
#                 df2.ilocp[i,j] = int(df2.iloc[i,j].replace(',', ''))          
  
  
#pandas를 numpy로 변경 후 저장
df1 = df1.values
df2 = df2.values
# print(type(df1), type(df2))
# print(df1.shape, df2.shape)    #(2220, 16) (1980, 16)

np.save('./_data/samsung.npy', arr=df1)
np.save('./_data/amore.npy', arr=df2)


#numpy 데이터 불러오기 
samsung = np.load('./_data/samsung.npy',allow_pickle=True)
amore = np.load('./_data/amore.npy', allow_pickle=True)

# print(samsung)
# print(amore)
print(samsung.shape)
print(amore.shape)


# dnn 구성하기 
def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column 

        if y_end_number > len(dataset): 
            break
        tmp_x = dataset[i:x_end_number, :] 
        tmp_y = dataset[x_end_number:y_end_number, 0] 
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x1, y1 = split_xy5(samsung, 5, 1) 
x2, y2 = split_xy5(amore, 5, 1) 
print(x2[0,:], "\n", y2[0])
# print(x2.shape)
# print(y2.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x1, y1, random_state=1, test_size= 0.3)
# print(x_train.shape)  #(1550, 5, 5)
# print(x_test.shape)    #(665, 5, 5)
# print(y_train.shape)    #(1550, 1)
# print(y_test.shape)     #(665, 1)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=1, test_size= 0.3)

# 2차원 reshape
x_train = np.reshape(x_train,
    (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test,
    (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
x2_train = np.reshape(x2_train,
    (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test,
    (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))
print(x2_train.shape)
print(x2_test.shape)

#####데이터 전처리####
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

scaler.fit(x2_train)
x2_train_scaled = scaler.transform(x2_train)
x2_test_scaled = scaler.transform(x2_test)
print(x_train_scaled[0, :])


x_train_scaled = np.reshape(x_train_scaled,
    (x_train_scaled.shape[0], 5, 5))
x_test_scaled = np.reshape(x_test_scaled,
    (x_test_scaled.shape[0], 5, 5))
x2_train_scaled = np.reshape(x2_train_scaled,
    (x2_train_scaled.shape[0], 5, 5))
x2_test_scaled = np.reshape(x2_test_scaled,
    (x2_test_scaled.shape[0], 5, 5))
print(x2_train.shape)
print(x2_test.shape)

#2 모델구성 
#2-1 모델 
input1 = Input(shape=(5, 5))
dense1 = LSTM(300, activation='relu', name='ds11')(input1)
dense2 = Dense(550, activation='relu', name='ds12')(dense1)
dense3 = Dense(730, activation='relu', name='ds13')(dense2)
output1 = Dense(850, activation='relu', name='ds14')(dense3)
#2-2 모델 2
input2 = Input(shape=(5, 5))
dense21 = LSTM(100, activation='linear', name='ds21')(input2)
dense22 = Dense(390, activation='linear', name='ds22')(dense21)
output2 = Dense(805, activation='linear', name='ds23')(dense22)
#2-3 모델병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name ='mg1')
merge2 = Dense(200, activation= 'relu', name ='mg2')(merge1)
merge3 = Dense(333, name ='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)       #여기서 1은 y를 뜻함 그래서 1개임
model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(patience=10)
modelcheckpoint = ModelCheckpoint(moniotr= 'val_loss', mode='auto', verbose=1, 
                                save_best_only=True, filepath='./_save/MCP/samsung_ensemble_MCP.hdf5')
model.fit([x_train_scaled, x2_train_scaled], y_train, validation_split=0.2, 
          verbose=1, batch_size=1, epochs=100, 
          callbacks=[early_stopping, modelcheckpoint])

loss, mse = model.evaluate([x_test_scaled, x2_test_scaled], y_test, batch_size=1)
print('loss : ', loss)
print('mse : ', mse)

y1_pred = model.predict([x_test_scaled, x2_test_scaled])


for i in range(5):
    print('시가 : ', y_test[i], '/ 예측가 : ', y1_pred[i])
# #3. 컴파일 훈련 
# model.compile(loss ='mse', optimizer='adam')
# model.fit([x_train_scaled, x2_train_scaled], y_train, epochs=10, batch_size=8)
# #4. 평가 예측
# loss = model.evaluate([x_test_scaled, x2_test_scaled], y_test)
# print('loss : ', loss)         


"""
loss :  7905110.0
mse :  7905110.0
시가 :  [154500.] / 예측가 :  [152129.39]
시가 :  [184500.] / 예측가 :  [182515.81]
시가 :  [234000.] / 예측가 :  [229976.66]
시가 :  [277000.] / 예측가 :  [280431.16]
시가 :  [173000.] / 예측가 :  [170556.48]

"""