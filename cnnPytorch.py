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
import skimage.io as io
from skimage.draw import circle

def createImage(w,h):
    img = np.ones((100,100))
    w = int(w/9)
    h = int(h/9)
    
    x1 = (100-w)/2
    x2 = (100+w)/2
    
    y1 = (100-h)/2
    y2 = (100+h)/2
    
    for i in range(100):
        for j in range(100):
            if(i in range(int(y1),int(y2)+1) and j in range(int(x1),int(x2)+1)):
                img[i,j] = 0.0
    
    
    return img


def createSquare(r):
    return createImage(2*r,2*r)


def createCircle(r):
    img = np.ones((100,100))
    r = int(r/9)
    rr, cc = circle(50,50, r)
    img[rr, cc] = 0
    return img

def saveImage(r,isCircle):
    
    if(isCircle):
        img = createCircle(r)
        fname = "dataset/circles/circle{}.jpg".format(r)
    
    else:
        img = createSquare(r)
        fname = "dataset/squares/square{}.jpg".format(r)
    
    
    
    
    img = img*255
    
    img = np.floor(img).astype('uint8')
    
    
    print("Saving {}".format(fname))    
    
    io.imsave(fname,img)



#class CNN(nn.Module):
#    def __init__(self):
#        super(CNN,self).__init__()
#        self.conv1 = nn.Conv2d(1,16,(3,3))
#        self.conv2 = nn.Conv2d(16,16,(3,3))
#        self.conv3 = nn.Conv2d(16,8,(3,3))
#        
#        self.pool = nn.MaxPool2d(2)
#        
#        self.fc1 = nn.Linear(8*10*10,200)
#        self.fc2 = nn.Linear(200,84)
#        self.fc3 = nn.Linear(84,2)
#        
#        
#    def forward(self,x):
#        x = self.conv1(x)
#        x = F.relu(x)
#        x = self.pool(x)
#        
#        x = self.conv2(x)
#        x = F.relu(x)
#        x = self.pool(x)
#        
#        x = self.conv3(x)
#        x = F.relu(x)
#        x = self.pool(x)
#        
#        x = x.view(-1, 8 * 10*10)
#        
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        
#        return x



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,8,(3,3))
        self.conv2 = nn.Conv2d(8,16,(3,3))
        self.conv3 = nn.Conv2d(16,32,(3,3))
        
        self.pool = nn.MaxPool2d(2,stride=2)
        
        self.fc1 = nn.Linear(14112,2)
#        self.fc2 = nn.Linear(200,84)
#        self.fc3 = nn.Linear(84,2)
        
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(x)
#        x = self.pool(x)
        
        x = x.view(-1, 14112)
        
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
        x = self.fc1(x)
        return x






#Channels first
#data = torch.randn(100,1,100,100)
#cnn = CNN()
#print(cnn(data).shape)




class CNNWrapper:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model = CNN().to(self.device)
    
    def generateData(self):
        circles = []
        squares = []
        
        
        for i in range(1,451):
            fNameCircle = 'dataset/circles/circle{}.jpg'.format(i)
            fNameSquare = 'dataset/squares/square{}.jpg'.format(i)
        
            circles.append([io.imread(fNameCircle,as_gray=True)])
            squares.append([io.imread(fNameSquare,as_gray=True)])
        
        
        circles = np.array(circles)
        squares = np.array(squares)
        
    
        circleTarget = np.array([0]*len(circles))
        squareTarget = np.array([1]*len(squares))
        
        
        X = np.concatenate((circles,squares),axis = 0)
        print(X.shape)
        Y = np.concatenate((circleTarget,squareTarget),axis=0)
        print(Y.shape)
        
        return X/255.0,Y

    
    def predict(self,r,isCircle):
        if(isCircle):
            image = createCircle(r)
        else:
            image = createSquare(r)
            
        
        x = torch.tensor([[image]]).float()
        
        
        
        m = nn.Softmax(dim=1)
        
        ans = m(self.model(x))
        
        classes = ['Circle','Square']
        index = torch.argmax(ans).cpu().numpy()
        return classes[index],ans[0][index].detach().cpu().numpy()*100
    
    
    def loadModel(self,fileName):
        device = torch.device('cpu')
        self.model = CNN()
        self.model.load_state_dict(torch.load(fileName, map_location=device))
    
    def saveModel(self,fileName):
        torch.save(self.model.state_dict(),fileName)
    
    def trainModel(self):
        X,Y = self.generateData()
        optimizer = optim.Adam(self.model.parameters())
#        optimizer = optim.SGD(self.model.parameters(),lr = 0.01)
        
        
        lossFunction = nn.CrossEntropyLoss()
        print(X)
        print(Y)
        for epoch in range(1000):
#            for x,y in zip(X,Y):
#                
#                
#                optimizer.zero_grad()
#                
#                pred = self.model(torch.FloatTensor([x]))
#                
#                
#                y = torch.LongTensor(y)
#                
#                output = lossFunction(torch.log(pred),y)
#                print(output)
#                output.backward()
#                optimizer.step()
            
            optimizer.zero_grad()
            output = self.model(torch.tensor(X).to(self.device).float())
            
            target = torch.tensor(Y).to(self.device).long()
            loss = lossFunction(output,target)
            print(epoch)
            print(loss)
            loss.backward()
            optimizer.step()
            
    
            
if __name__ == "__main__":
           
    trainer = CNNWrapper()
# #trainer.generateData()

#  #print(trainer.model(torch.tensor([[1,2],[2,3]]).float()))
    trainer.loadModel('savedModels/FinalCNN.pt')
    
#trainer.predict(10,True)