import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

class data_all():
    def __init__(self, root_path):
        self.root_path=root_path
        self.data_path=os.path.join(self.root_path, 'data')
        self.label_path=os.path.join(self.root_path, 'label')

    def __getitem__(self, index):
        pic_list=os.listdir(self.data_path)
        pic_name=pic_list[index]
        pic_path=os.path.join(self.data_path, pic_name)
        pic=Image.open(pic_path).convert('L')
        label_list=os.listdir(self.label_path)
        label_name=label_list[index]
        label_path=os.path.join(self.label_path, label_name)
        with open(label_path, 'r') as file:
            content = file.read()
        return pic,float(content)
    
    def __len__(self):
        return len(os.listdir(self.data_path))

def r_square(data,label,model,types):
    pred=model.predict(data)
    if(types==0):
        1
    elif(types==1):
        pred=np.argmax(pred,axis=1)
        label=np.argmax(label,axis=1)
    mean_observed = np.mean(label)
    total_variation = np.sum((label - mean_observed) ** 2)
    residuals = label - pred.flatten()
    residual_sum_of_squares = np.sum(residuals ** 2)
    r_squared = 1 - (residual_sum_of_squares / total_variation)
    print("R-squared:", r_squared)
    
    paired_values = list(zip(label, pred))
    sorted_values = sorted(paired_values)
    sorted_x = [pair[0] for pair in sorted_values]
    sorted_y = [pair[1] for pair in sorted_values]
    sorted_x = np.array(sorted_x).flatten()
    sorted_y = np.array(sorted_y).flatten()

    coefficients = np.polyfit(sorted_x, sorted_y, 1)
    poly_function = np.poly1d(coefficients)

    plt.scatter(sorted_x,sorted_y, color='b', label='Actual') 
    plt.plot(sorted_x, poly_function(sorted_x), color='r', label='Fitted')  
    plt.plot(sorted_x,sorted_x, color='g', label='x=y')
    plt.xlabel('true')
    plt.ylabel('pred')
    plt.legend()
    plt.show()
