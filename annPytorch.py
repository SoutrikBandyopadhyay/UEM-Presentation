# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:28:11 2019

@author: Paramita2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



class ANN(nn.Module):
    def __init__(self):
        super(ANN,self).__init__()
        self.fc1 = nn.Linear(2,10)
        self.fc2 = nn.Linear(10,2)
        self.smax = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x







class ANNWrapper:
    def __init__(self):
        self.model = ANN()
    
    def generateData(self):
        w = np.zeros((900,1))
        h = np.zeros((900,1))
        
        x_train_s=np.zeros((900,2))
        x_train_r=np.zeros((900,2))    
    
        x_train=np.zeros((1800,2))
        x_target=np.zeros((1800))    
    
        poss = range(5,901,5)
        for i in range(1,900):
            w[i] = np.random.choice(poss)
            h[i] = np.random.choice(poss)        
        x_train_r=np.concatenate((w,h),axis=1)    
        for j in range(900):
            if x_train_r[j,0]==x_train_r[j,1]:
                x_train_r[j,0]=x_train_r[j,0]+1
        for j in range(900):
            x_train_s[j,0]=j+1
            x_train_s[j,1]=j+1    
        x_train=np.concatenate((x_train_r,x_train_s),axis=0)
        for j in range(1800):
            if j<900:        #Rectangle
                x_target[j]=1
            else:              #Square
                x_target[j]=0
        return x_train,x_target
    
    def predict(self,w,h):
        x = torch.tensor([w,h]).float()
        m = nn.Softmax(dim=0)
        
        ans = m(self.model(x))
        classes = ['Square','Rectangle']
        index = torch.argmax(ans).cpu().numpy()
        return classes[index],ans[index].detach().cpu().numpy()*100
    
    
   
    
    def trainModel(self):
        X,Y = self.generateData()
        optimizer = optim.Adam(self.model.parameters())
        
        lossFunction = nn.CrossEntropyLoss()
               
        for epoch in range(1000):
           
            optimizer.zero_grad()
            output = self.model(torch.tensor(X).float())
            
            target = torch.tensor(Y).long()
            loss = lossFunction(output,target)
            print(epoch)
            print(loss)
            loss.backward()
            optimizer.step()
            
    
    
    
    
    
    
    
    
    def loadModel(self,fileName):
        self.model = torch.load(fileName)
    
    def saveModel(self,fileName):
        torch.save(self.model,fileName)
            
            
