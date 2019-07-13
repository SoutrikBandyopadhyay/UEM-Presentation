# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:15:31 2019

@author: Paramita2
"""

import pygame
import numpy as np

from cnnPytorch import CNNWrapper,CNN

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
pygame.display.set_caption('Circle Or Square ?')

#TIMING
clock = pygame.time.Clock()



crashed = False

r = 250
isCircle = True

deltaR = 0

changeAmount = 5
predict = False

locked = False
prediction = ''
cnn = CNNWrapper()
cnn.loadModel('savedModels/FinalCNN.pt')

classNames = ['Circle','Square']
confidence = ""
controls = pygame.image.load('controlsCircle.png')

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
                
                deltaR = -changeAmount
                
            if event.key == pygame.K_RIGHT:
                deltaR = changeAmount
                
            
            if event.key == pygame.K_a:
                r-=changeAmount

            if event.key == pygame.K_d:
                r+=changeAmount


            if event.key == pygame.K_ESCAPE:
                crashed = True

            if event.key == pygame.K_t:
                measureDisplay = not measureDisplay

            if event.key == pygame.K_l:
                isCircle = not isCircle 
            if event.key == pygame.K_p:
                
                prediction,confidence = cnn.predict(r,isCircle)

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                deltaR = 0
            if event.key == pygame.K_RIGHT:
                deltaR = 0

        
    gameDisplay.fill(white)

    


    # height += deltaH
    # width += deltaW
    # print(deltaR)
    r += deltaR
    # print(r)
    limit = int(np.ceil(0.20 * displayWidth/changeAmount)*changeAmount)

    r = max(0,r)
    r = min(r,limit)

    # height = max(0,height)
    # height = min(height,900)

    # width = max(0,width)
    # width = min(width,900)
    
    # if(locked):
    #     height = width
    


    # print(width,height)


    midX = displayWidth/2
    midY = displayHeight/2

    x = midX - r
    y = midY - r


    font = pygame.font.Font('freesansbold.ttf',int(0.02*displayHeight))

    gameDisplay.blit(controls,(0,0))
    if(not isCircle):
        if(measureDisplay):
            widthText = font.render("{} px".format(2*r) , True, black)
            widthTextRect = widthText.get_rect()
            widthTextRect.center = (midX,y-20)

            gameDisplay.blit(widthText,widthTextRect)

        pygame.draw.rect(gameDisplay,green,[x,y,2*r,2*r])
    else:
        if(measureDisplay):
            widthText = font.render("{} px".format(2*r) , True, black)
            widthTextRect = widthText.get_rect()
            widthTextRect.center = (midX,y-20)

            gameDisplay.blit(widthText,widthTextRect)

        pygame.draw.circle(gameDisplay,green,(int(midX),int(midY)),r)

        
    # heightText = font.render("{} px".format(height) , True, black)
    # heightTextRect = heightText.get_rect()
    # heightTextRect.center = (x-50,midY)

    # gameDisplay.blit(heightText,heightTextRect)

    if(confidence):
        confidence = int(confidence)
        predText = font.render("Prediction = {}({}%)".format(prediction,confidence),True,black)
    else:
        predText = font.render("",True,black)

    predTextRect = predText.get_rect()
    predTextRect.center = (midX , int(0.9*displayHeight))

    gameDisplay.blit(predText,predTextRect)


    
    pygame.display.update()
    clock.tick(120)    

pygame.quit()
quit()








#
#
#pygame.quit()