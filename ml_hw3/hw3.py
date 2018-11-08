import sys
import numpy as np
#import keras
import hw3_func as func
import pandas as pd 
import matplotlib.pyplot as plt
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap='binary')
    plt.savefig("x.png")




def main(args):
    file_name=sys.argv[1]

    file_name = "C:/Users/cyhu/pythonwork/data/ML_hw3/train.csv"
    train_data = pd.read_csv(file_name)
    print(0)
    X,Y = func.data_process_init (train_data)
    print(1)
    X = func.feat_init(X)
    print(2)    

    print(X)
    plot_image(X[0])
if __name__ == '__main__':
    main(sys.argv)


