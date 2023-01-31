#가위 바위 보 모델 만들기 
import numpy as np  
import tensorflow as tf                   
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(ds_train, ds_test), ds_info = tfds.load(
    'rock_paper_scissors',
    split=['train', 'test'],
    # shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

train_datagen = ImageDataGenerator(
    rescale=1./255)

test_datagen = ImageDataGenerator(
    rescale= 1./255)     

pred_datagen = ImageDataGenerator(
    rescale= 1./255)     

xy_train = train_datagen.flow_from_directory(
    './_data/rps/',
    target_size=(64, 64), #증폭됨 기존 사이즈가 150,150이니깐
    batch_size=100,       #배치를 크게 잡아주면 총 데이터수를 알 수 있게 된다. 
    class_mode='categorical',
    # class_mode='binary',   # binary가 아닌 categoricaldms 원핫이 된것. 
    color_mode='rgb', 
    shuffle=True
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    './_data/rps/',
    target_size=(64, 64), #증폭됨 기존 사이즈가 150,150이니깐
    batch_size=100,
    # class_mode='binary',
    class_mode='categorical',
    color_mode='rgb', 
    shuffle=True
    # Found 120 images belonging to 2 classes.
)


xy_pred = pred_datagen.flow_from_directory(
    './_data/rps/',
    batch_size=100,
    target_size=(64, 64),
    # class_mode='categorical',
    class_mode='categorical',
    color_mode='rgb')
    
    
    
print(xy_train, xy_test, xy_pred)            


# np.save('./_data/rps/rps_train.npy', arr=xy_train[0][0])
# np.save('./_data/rps/rps_test.npy', arr=xy_train[0][1])
# np.save('./_data/rps/rps_pred.npy', arr=xy_train[0][0])
