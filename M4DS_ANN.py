#!/usr/bin/env python
# coding: utf-8

# In[723]:


import numpy as np
import pandas as pd


# In[733]:


class Dense_Layer:
    #Layer initialization
    def __init__(self,inputsize,neurons):
        #Initilize weights and biases
        self.weights = 0.01 * np.random.randn(inputsize, neurons)
        self.biases = np.zeros((1, neurons))
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.inputs=inputs;
        #print('input to neuron',self.inputs)
        self.output = np.dot(inputs, self.weights) + self.biases
        #print('output of neuron',self.output)
    def backward(self,dvalues):
        self.dweights=np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)       


# In[734]:


# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.inputs=inputs
        self.output = np.maximum(0, inputs)
        
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        #print('start')
        #print(dvalues)
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        #print('middle')
        #print(self.dinputs)


# In[735]:


#Sigmpoid actiation
class Activation_Sigmoid:
    # Forward Pass
    def forward(self,inputs):
        self.inputs=inputs
        self.output=1/(1+np.exp(-inputs))
        #print('sigmoid output',self.output)
    
    def backward(self, dvalues,original):
        #print('start')
        #print(dvalues)
        self.dinputs=dvalues.copy()
        self.dinputs=(self.dinputs*(original-dvalues))/len(dvalues)
        #print('middle')
        #print(self.dinputs)


# In[736]:


class Loss:
    def calculate(self,y_true, y_predicted):
        return np.sqrt(np.sum((y_true-y_predicted)**2)/len(y_true))


# In[737]:


class Optimizer_SGD:
    # Initialize optimizer -set settings
    #learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=0.05):
        self.learning_rate=learning_rate
        
    # Update parameters
    def update_params(self,layer):
        #print('original',layer.weights,'new',layer.dweights)
        layer.weights -=-self.learning_rate * layer.dweights
        layer.biases -= -self.learning_rate * layer.dbiases
        #print(layer.dweights)
        


# In[738]:


X=pd.read_csv('E:\M4DS\ANNX.csv')


# In[739]:


Y=pd.read_csv('E:\M4DS\ANNY.csv')


# In[740]:


dense1=Dense_Layer(2,5)
dense2=Dense_Layer(5,4)
dense3=Dense_Layer(4,3)
dense4=Dense_Layer(3,2)
dense5=Dense_Layer(2,1)


activation1=Activation_Sigmoid()
activation2=Activation_Sigmoid()
activation3=Activation_Sigmoid()
activation4=Activation_Sigmoid()
activation5=Activation_Sigmoid()


Lossfunction=Loss();
optimizer = Optimizer_SGD()

    
 


# In[741]:


for epoch in range(100000):
    dense1.forward(np.array(X))
    activation1.forward(dense1.output)
    
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    dense4.forward(activation1.output)
    activation2.forward(dense2.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    predictions=activation2.output.copy()
    predictions[predictions>0.3]=1
    predictions[predictions<0.3]=0
    accuracy = np.mean(predictions==np.array(Y))
    loss=Lossfunction.calculate(np.array(Y),activation2.output)
    
    activation2.backward(activation2.output,np.array(Y))
    dense2.backward(activation2.dinputs)
    #print(activation2.dinputs)
    
    activation1.backward(dense2.dinputs,np.array(Y))
    dense1.backward(activation1.dinputs)
    
    optimizer.update_params(dense2)
    optimizer.update_params(dense1)
    
    #print(dense2.weights)
    #print(dense2.dweights)
    if not epoch % 1000:
        print(f'epoch: {epoch}, ' +f'acc: {accuracy:.3f}, ' +f'loss: {loss:.3f}')


# In[ ]:





# In[ ]:




