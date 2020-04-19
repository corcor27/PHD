import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from scipy.stats import norm

iris = datasets.load_iris()
X = iris.data[:, :]
y = iris.target


def mean(values):
    # calculates mean of array
    val = sum(values)/float(len(values))
    return val

def std(values):
    # calculates std of array
    std = np.std(values)
    return std
    
def summerys(array, class_list, rr, kk):
    # calculates mean and std of array
    list0 = []
    for ii in range(0, len(class_list)):
        if class_list[ii] == rr:
            list0.append(array[ii,:])
    yy = np.array(list0)        
    list = (yy[:,kk]) 
    mean1 = sum(list)/ len(list)
    std1 = std(list)
    return mean1, std1




def calculate_prob(x,mean,std):
    # calculate values based on gaussian distrabution
    function = np.exp(-((x-mean)**2 / (2 * std**2 )))
    return (1 / (np.sqrt(2 * np.pi) * std)) * function

def calculateClassProbability(train_class):
    probabilities = []
    for ii in range(0,3):
        count = 0 
        for tt in range(0,len(train_class)):
            if train_class[tt] == ii:
                count += 1
        gg = count/len(train_class)
        probabilities.append(gg)
        
    return probabilities

def test_train_sets(list_data, listofclases):
    X = list_data
    y = listofclases
    
    train0 = X[0:40]
    test0 = X[40:50]
    train1 = X[50:90]
    test1 = X[90:100]
    train2 = X[100:140]
    test2 = X[140:150]  
    y0train = y[0:40]
    y0test = y[40:50]
    y1train = y[50:90]
    y1test = y[90:100]
    y2train = y[100:140]
    y2test = y[140:150]
    train_X = []
    test_X = []
    train_y = []
    test_y = []
    train_X.extend(train0)
    train_X.extend(train1)
    train_X.extend(train2)
    test_X.extend(test0)
    test_X.extend(test1)
    test_X.extend(test2)
    train_y.extend(y0train)
    train_y.extend(y1train)
    train_y.extend(y2train)
    test_y.extend(y0test)
    test_y.extend(y1test)
    test_y.extend(y2test)
    return np.array(train_X), np.array(test_X), np.array(train_y), np.array(test_y)
 



def class_proablity(X, y):
    ff = test_train_sets(X,y)
    train_X = ff[0]
    test_X = ff[1]
    train_y = ff[2]
    test_y = ff[3]
    correct = 0
    wrong = 0
    for zz in range(0, len(test_y)):
        prob = []
        for rr in range(0,3):
            val = calculateClassProbability(train_y)[rr]
            for kk in range(0, 4):
                mean, std = summerys(train_X, train_y, rr, kk)
                gg = calculate_prob(test_X[zz,kk],mean,std)
                val *= gg
            prob.append(val)

        
        hh = np.argmax(np.array(prob))
        
        if hh == test_y[zz]:
            correct += 1
        else:
            wrong += 1
    correct_precentage = (correct/ len(test_y))*100
    
    return correct_precentage

tt = class_proablity(X, y)
print('precentage correct {}%'.format(tt))

        
            
        
    



    