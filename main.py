
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import math

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import preprocessing
from numpy import exp,reshape
from matplotlib import pyplot as plt

#calculating loss for training database
def calc_training_data_loss(thetaT, X_train_minmax ,bias):

    # calculating hypothesis function
    z = np.dot(thetaT , X_train_minmax.T) + bias           # (1,455)

    #calculating sigmoid function
    sigmoid_function = 1 / (1+ exp(-z))                    # (1,455)

    #calculating cost function
    summation_cost_func =   np.dot(np.log(sigmoid_function) , y_train ) + np.dot( np.log(1 - sigmoid_function) , (1 - y_train ) ) #(1,1)

    #calculating loss
    loss= np.dot ( (-1/ X_train_minmax.shape[0]) , (summation_cost_func) )
    #print("loss in training dataset : " ,loss)
    
    return [sigmoid_function,loss]

#calculating loss for validation database
def calc_validation_data_loss(thetaT, X_val_minmax, bias ):
    
    # calculating hypothesis function
    z = np.dot(thetaT , X_val_minmax.T) + bias           # (1,455)

    #calculating sigmoid function
    sigmoid_function = 1 / (1+ exp(-z))                 # (1,455)

    #calculating cost function
    summation_cost_func =   np.dot(np.log(sigmoid_function) , y_val ) + np.dot( np.log(1 - sigmoid_function) , (1 - y_val ) ) #(1,1)

    #calculating loss
    loss= np.dot( (-1/ X_val_minmax.shape[0]) , (summation_cost_func) )
    #print("loss in validation dataset : " ,loss)

    return [sigmoid_function,loss]

#applying gradient descent
def calc_weights(learning_rate, theta, m, y ,sigmoid_func,feature_x):
    x=np.array(y.T)- sigmoid_func
    delta_theta= (-1 / m ) * np.dot(x , np.array(feature_x) )
    theta=theta - np.dot(learning_rate , delta_theta)
    return theta

#applying gradient descent
def calc_bias(learning_rate, bias, m, y, sigmoid_func):
    x=np.array(y.T)- sigmoid_func
    delta_bias= (-1 / m ) * x
    bias=bias - np.sum(np.dot(learning_rate , delta_bias))
    return bias

#reading data from the csv file
my_data= pd.read_csv('wdbc.csv',header=None, delimiter=',')

#converting csv to dataframe
df=pd.DataFrame(my_data)

#replacing B->0 and M->1 in target set
modDfObj= df.replace(to_replace ="B",value =0)
modDfObj= modDfObj.replace(to_replace ="M",value =1)

#drop the id column
modDfObj = modDfObj.drop([df.columns[0]] ,  axis='columns')

#slicing the data into x and y
y=modDfObj.iloc[:,0]                   # shape of y (569,1)
x=modDfObj.iloc[:, 1:]                 # shape of x (569, 30)
 
#split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False) 
# training (455, 30) (455,1) Testing data (114, 30) (114,1)


#split the traning dataset into validation and testing data
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)   
#Testing data (57, 30) (57,1) Validation data (57, 30) (57,1)

#min-max normalization
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.fit_transform(X_test)
X_val_minmax = min_max_scaler.fit_transform(X_val)

# initially creating theta
theta=np.zeros((30, 1))                             # (30,1)

# initially calculating bias (y=mx+c)
bias=0

training_loss_data=[]
validation_loss_data=[]

learning_rate=0.1
    
#start running loops
#3000 gives a po=roper graph
for x in range(0,1900):

    #theta transpose
    thetaT= theta.T                                     # (1,30)

    training_loss_val = calc_training_data_loss(thetaT, X_train_minmax ,bias)
    #training_loss_val[0]  #sigmoid func
    #training_loss_val[1]  #loss
    training_loss_data.append(training_loss_val[1])

    validation_loss_val = calc_validation_data_loss(thetaT, X_val_minmax, bias)
    validation_loss_data.append(validation_loss_val[1])

    
    #changing theta for training dataset
    m_training=X_train_minmax.shape[0]
    for i in range(0,len(theta)):
        theta[i] = calc_weights(learning_rate,theta[i], m_training , y_train, training_loss_val[0] , X_train_minmax[:,i])
   
    bias = calc_bias(learning_rate, bias, m_training, y_train, training_loss_val[0])
  


# In[5]:


plt.plot(range(0,1900), training_loss_data)
plt.plot(range(0,1900), validation_loss_data)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["training set","validation set"])


# In[5]:


#Testing with the testing dataset
zz = np.dot(X_test_minmax ,theta ) + bias
y = 1 / (1+ exp(-zz)) 
print(type((y)))

#classifying predicted result to class 0 and class 1
for i in range(0,len(y)):
    if y[i]<0.5:
        y[i]=0
    else:
        y[i]=1
      
 #identifying the count of true positives, true negatives, false positives and false negatives.
TP=TN=FP=FN=0
y_test=np.array(y_test)

for i in range(0,len(y)):
    if(y[i]==y_test[i] and y[i]==1):
        TP+=1
    elif(y[i]==y_test[i] and y[i]==0):
        TN+=1
    elif(y[i]!=y_test[i] and y[i]==1):
        FP+=1
    elif(y[i]!=y_test[i] and y[i]==0):
        FN+=1
        
print(TP,TN,FP,FN) 

accuracy= (TP+TN )/(TP+TN+FP+FN) 
precision= TP / (TP+FP)   
recall= TP/(TP+FN)

print("Accuracy : " + str(accuracy))
print("Precision : " + str(precision))
print("Recall : " + str(recall))

