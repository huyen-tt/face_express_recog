import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os

def plot_confusion_matrix(cm, labels_name, title, acc):
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    # plt.colorbar()
    num_class = np.array(range(len(labels_name)))
    plt.xticks(num_class, labels_name, rotation=90)
    plt.yticks(num_class, labels_name)
    plt.ylabel('Target')
    plt.xlabel('Prediction')
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.tight_layout()
    plt.savefig(os.path.join('./Confusion_matrix/raf-db', "fer_acc" + str(acc) + ".png"), format='png')
    plt.show()
    
if __name__ == "__main__":  
    predict = pd.read_csv('models/predict_fer_res.csv', header=None)[1]
    target = pd.read_csv('models/target_fer-res.csv', header=None)[1]
    predictValue = predict.values
    targetValue  = target.values
    cm = confusion_matrix(targetValue, predictValue)

    labels_name= ['Ngạc nhiên','Sợ hãi','Ghê tởm','Hạnh phúc','Buồn bã','Tức giận','Trung tính']
    title= 'Confusion Matrix'
    acc= 0.6994
    plot_confusion_matrix(cm, labels_name= labels_name, title= title, acc= acc)


    

