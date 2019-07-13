# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:15:31 2019

@author: Paramita2
"""

import pygame
import numpy as np
# from cnnTraining import CNN,createImage
from annPytorch import ANNWrapper,ANN

pygame.init()

infoObject = pygame.display.Info()


displayWidth = infoObject.current_w
displayHeight = infoObject.current_h


full = True
if full:
    gameDisplay = pygame.display.set_mode((displayWidth,displayHeight),pygame.FULLSCREEN)
else:
    gameDisplay = pygame.display.set_mode((displayWidth,displayHeight))

#Color
black = (0,0,0)
white = (255,255,255)
blue = (86, 145, 240)
green = (22, 217, 90)


#Title
pygame.display.set_caption('Rectangle Or Square ?')

#TIMING
clock = pygame.time.Clock()



crashed = False


height = 500
width = 500

deltaH = 0
deltaW = 0

changeAmount = 5
predict = False

locked = False
prediction = ''
# cnn = CNN()
# cnn.loadModel('savedModels/cnn_15_0.82.hdf5')

ann = ANNWrapper()
ann.loadModel('savedModels/bestPytorchANN.pt')




confidence = ''
controls = pygame.image.load('controlsRect.png')
size = int(displayWidth *0.30)
size2 = int(displayWidth *0.35)



controls = pygame.transform.scale(controls,(size,size2))



measureDisplay = False

while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                deltaW = -changeAmount
            if event.key == pygame.K_RIGHT:
                deltaW = changeAmount
            if event.key == pygame.K_UP:
                deltaH = changeAmount
            if event.key == pygame.K_DOWN:
                deltaH = -changeAmount
            if event.key == pygame.K_w:
                height += changeAmount
            if event.key == pygame.K_s:
                height += -changeAmount
            if event.key == pygame.K_a:
                width += -changeAmount
            if event.key == pygame.K_d:
                width += changeAmount
            if event.key == pygame.K_ESCAPE:
                crashed = True
            if event.key == pygame.K_l:
                locked = not locked 
            if event.key == pygame.K_t:
                measureDisplay = not measureDisplay
                               
            if event.key == pygame.K_p:
                # print("Here")
                # prediction = classNames[0]
                # if width == height:
                #     key = 1
                # else:
                #     key = 0

                prediction,confidence = ann.predict(width,height)






        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or pygame.K_RIGHT:
                deltaW = 0
            if event.key == pygame.K_UP or pygame.K_DOWN:
                deltaH = 0

        
    gameDisplay.fill(white)

    


    height += deltaH
    width += deltaW


    limit = int(np.ceil(0.40 * displayWidth/changeAmount)*changeAmount)

    height = max(0,height)
    height = min(height,limit)

    width = max(0,width)
    width = min(width,limit)
    
    if(locked):
        height = width
    

    # print(width,height)



    midX = displayWidth/2
    midY = displayHeight/2

    x = midX - width/2
    y = midY - height/2




    font = pygame.font.Font('freesansbold.ttf',int(0.02*displayHeight))
    
    gameDisplay.blit(controls,(0,0))
    if(measureDisplay):
        widthText = font.render("{} px".format(width) , True, black)
        widthTextRect = widthText.get_rect()
        widthTextRect.center = (midX,y-20)

        gameDisplay.blit(widthText,widthTextRect)

        heightText = font.render("{} px".format(height) , True, black)
        heightTextRect = heightText.get_rect()
        heightTextRect.center = (x-50,midY)

        gameDisplay.blit(heightText,heightTextRect)
    
    if(confidence):
        confidence = int(confidence)
        predText = font.render("Prediction = {}({}%)".format(prediction,confidence),True,black)
    else:
        predText = font.render("",True,black)
    
    predTextRect = predText.get_rect()
    predTextRect.center = (midX , int(0.9*displayHeight))

    gameDisplay.blit(predText,predTextRect)


    pygame.draw.rect(gameDisplay,blue,[x,y,width,height])
    
    pygame.display.update()
    clock.tick(120)    

pygame.quit()
quit()








#
#
#pygame.quit()