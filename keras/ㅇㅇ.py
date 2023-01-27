from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

model = Sequential()
model.add(SimpleRNN(4, input_shape=(2,3))) 
# model.add(SimpleRNN(4, input_length=2, input_dim=3)) 와 동일하다 
model.summary()


