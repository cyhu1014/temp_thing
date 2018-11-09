##ntuee machine learning hw3 test file


import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import pandas as pd 
import hw3_func_test as func




def main(args):
    #file_name=sys.argv[1]

    file_name = sys.argv[1]
    train_data = pd.read_csv(file_name)
    X = func.data_process_init (train_data,label=False)
    X = func.feat_init(X)
    X_4D =X.reshape(X.shape[0],48,48,1).astype('float32')
    X_4D_normalize =X_4D/255
    print("data_init_done")
    #CNN
    model = Sequential()
    ##
    model.add(Conv2D(filters=64,
                    kernel_size=(5,5),
                    padding='same',
                    input_shape=(48,48,1),
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    ##
    model.add(Conv2D(filters=128,
                    kernel_size=(3,3),
                    padding='same',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    ##
    model.add(Conv2D(filters=512,
                    kernel_size=(3,3),
                    padding='same',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.35))
    ##
    model.add(Conv2D(filters=512,
                    kernel_size=(3,3),
                    padding='same',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7,activation='softmax'))
    print(model.summary())
    model.load_weights("testmodel.h5")
    prediction = model.predict_classes(X_4D_normalize)
    
    ###create summit file
    test_label=[]
    test_title =[]
    test_title.append("id")
    test_label.append("label")
    for i in range (len(prediction)):
        test_title.append(str(i))
        test_label.append(prediction[i])
    df =pd.DataFrame(test_label,test_title)
    df.to_csv(sys.argv[2],header=False)
if __name__ == '__main__':
    main(sys.argv)

