import tensorflow as tf                  
print(tf.__version__)   #2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)  #gpu로 돌렸을 때 [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

if(gpus): 
    print('쥐피유 돈다')
else:
    print('쥐피유 안돈다')    #if랑 else는 한세트임 
