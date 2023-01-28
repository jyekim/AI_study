import numpy as np                      
import pandas as pd                    

# #numpy 데이터 불러오기 
# samsung = np.load('./_data/samsung.npy')
# amore = np.load('./_data/amore.npy')

# print(samsung)
# print(amore)
# print(samsung.shape)
# print(amore.shape)


#1. 데이터 
path = './_data/samsung/'
df1 = pd.read_csv(path + 'amore.csv', index_col=0,
                  header=0, encoding='cp949', sep=',',thousands=',')
print(df1)
print(df1.shape)    #(1980, 16)

df2 = pd.read_csv(path + 'samsung.csv', index_col=0,
                  header=0, encoding='cp949', sep=',', thousands=',')
print(df2)
print(df2.shape)   #(2221, 16)
print(df1.dtypes)

# # 삼성전자의 모든 데이터
# for i in range(len(df1.index)):       # 거래량 str 을 int 변경
#          for j in range(len(df1.iloc[i])):
#                 df1.iloc[i,j] = int(df1.iloc[i,j].replace(',', ''))
# # 아모레의 모든 데이터
# for i in range(len(df2.index)):
#          for j in range(len(df2.iloc[i])):
#                 df2.ilocp[i,j] = int(df2.iloc[i,j].replace(',', ''))          


#일자 오름차순(최근날짜를 가장 아래로)
df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True]) 
print(df1)
print(df2)
  
  
#pandas를 numpy로 변경 후 저장
df1 = df1.values
df2 = df2.values
print(type(df1), type(df2))
print(df1.shape, df2.shape)    #(2220, 16) (1980, 16)

np.save('./_data/samsung.npy', arr=df1)
np.save('./_data/amore.npy', arr=df2)


#numpy 데이터 불러오기 
samsung = np.load('./_data/samsung.npy',allow_pickle=True)
amore = np.load('./_data/amore.npy', allow_pickle=True)

print(samsung)
print(amore)
print(samsung.shape)
print(amore.shape)

# dnn 구성하기 
def split_xy5(datasets, time_steps, y_column):
    x, y =list(), list()
    for i in range(len(datasets)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        
        if y_end_number > len(datasets):
            break
        tmp_x = datasets[i:x_end_number, :]
        tmp_y = datasets[x_end_number:y_end_number, 5]
        x.append(tmp_x)
        y.append(tmp_y)
        return np.array(x), np.array(y)
    x, y = split_xy5(samsung, 1, 1)
    print(x[0,:], "\n", y[0])
    print(x.shape)
    print(y.shape)

# x1_datasets = np.array([range(100), range(301,401)]).transpose()
# print(x1_datasets.shape)   #(2, 100)  .transpose()한 후에 (100, 2)   #삼성전자의 시가, 고가
# x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).T
# print(x2_datasets.shape)   #(100, 3)                                  #아모레의 시가, 고가 , 종가


# y = np.array(range(2001, 2101))  #(100, )     #삼성전자의 하루 뒤 종가 

# from sklearn.model_selection import train_test_split
# x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
#     x1_datasets, x2_datasets, y, train_size=0.7, random_state=1234
# )

# print(x1_train.shape, x2_train.shape, y_train.shape)   #(70, 2) (70, 3) (70,)
# print(x2_test.shape, x2_test.shape, y_test.shape)      #(30, 3) (30, 3) (30,)


"""
#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델 

input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

#2-2 모델 2
input2 = Input(shape=(3,))
dense21 = Dense(21, activation='linear', name='ds21')(input2)
dense22 = Dense(22, activation='linear', name='ds22')(dense21)
output2 = Dense(23, activation='linear', name='ds23')(dense22)

#2-3 모델병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name ='mg1')
merge2 = Dense(12, activation= 'relu', name ='mg2')(merge1)
merge3 = Dense(13, name ='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)#여기서 1은 y를 뜻함 그래서 1개임

model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()

#3. 컴파일 훈련 
model.compile(loss ='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=8)

#4. 평가 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss)                                                           """