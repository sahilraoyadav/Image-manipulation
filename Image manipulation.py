import cv2
import numpy as np
import math as m


def main(i=0):
    #Select file
    choice = i
    
    if choice ==0:
        arrowScale = 500
        arrowDensity = 10
        window = 70
        file1 = "mayaSolidShapes1.png"
        file2 = "mayaSolidShapes2.png"
    if choice ==1:
        arrowScale = 500
        arrowDensity = 5
        window = 10
        file1 = "Paul1.jpg"
        file2 = "Paul2.jpg"
    if choice ==2:
        arrowScale = 50
        arrowDensity = 10
        window = 10
        file1 = "sphere1.jpg"
        file2 = "sphere2.jpg"
    if choice ==3:
        arrowScale = 500
        arrowDensity = 5
        window = 3
        file1  ="tree1.jpg"
        file2  ="tree2.jpg"
    if choice ==4:
        arrowScale = 500
        arrowDensity = 10
        window = 70
        file1 = "mayaShapes1.png"
        file2 = "mayaShapes2.png"
    if choice ==5:
        arrowScale = 500
        arrowDensity = 10
        window = 10
        file1 = "house1.bmp"
        file2 = "house2.bmp"

    # Read file
    image1 = cv2.imread(file1)
    image2 = cv2.imread(file2)

    # Shape
    sizex = len(image1[0])
    sizey = len(image1)

    # convert to grayscale
    grayIm1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    grayIm2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
   
    # gradient in X,Y direction
    gradImX = cv2.Sobel(grayIm1,cv2.CV_64F,1,0,ksize=3)
    gradImY = cv2.Sobel(grayIm1,cv2.CV_64F,0,1,ksize=3)
    
    # Sum Matrix
    ixxSum = np.zeros((sizey, sizex ,1),np.float32)
    iyySum = np.zeros((sizey, sizex ,1),np.float32)
    ixySum = np.zeros((sizey, sizex ,1),np.float32)
    ixtSum = np.zeros((sizey, sizex ,1),np.float32)
    iytSum = np.zeros((sizey, sizex ,1),np.float32)

    # Gradient
    ixx = np.zeros((sizey, sizex ,1),np.float32)
    iyy = np.zeros((sizey, sizex ,1),np.float32)
    ixy = np.zeros((sizey, sizex ,1),np.float32)
    iyt = np.zeros((sizey, sizex ,1),np.float32)
    ixt = np.zeros((sizey, sizex ,1),np.float32)
    itt =  np.zeros((sizey, sizex ,1),np.float32)
    for i in range(sizey):
        for j in range(sizex):
            itt[i][j] = float(grayIm2[i][j])-float(grayIm1[i][j])
            
    # For holding ang,mag,u,v
    polar = np.zeros((sizey, sizex ,4),np.float32)
    
    for i in range(sizey):
        for j in range(sizex):
            ixx[i][j] = gradImX[i][j] * gradImX[i][j]
            iyy[i][j] = gradImY[i][j] * gradImY[i][j]
            ixy[i][j] = gradImX[i][j] * gradImY[i][j]
            ixt[i][j] = gradImX[i][j] * itt[i][j]
            iyt[i][j] = gradImY[i][j] * itt[i][j]

    
    maxMag = 0
    for i in range(window, sizey-window):
        for j in range(window, sizex-window):
            ixxSum = np.sum(ixx[i - window : i + 1 + window , j - window : j + 1 + window])
            iyySum = np.sum(iyy[i - window : i + 1 + window , j - window : j + 1 + window])
            ixySum = np.sum(ixy[i - window : i + 1 + window , j - window : j + 1 + window])
            ixtSum = np.sum(ixt[i - window : i + 1 + window , j - window : j + 1 + window])
            iytSum = np.sum(iyt[i - window : i + 1 + window , j - window : j + 1 + window])
    
            det = (ixxSum * iyySum) - (ixySum**2)
            if det == 0:
                continue

            u = ( (ixtSum * iyySum) - (ixySum * iytSum)) * (-1/det)
            v = (-(ixySum * ixtSum) + (ixxSum * iytSum)) * (-1/det)
            
            ang = m.atan2(v,u)
                
            mag = m.sqrt((u**2) + (v**2))
            if mag>maxMag:
                maxMag = mag
            polar[i][j] = [ang,mag,u,v]
            
    anglePic = np.zeros((sizey, sizex ,3),np.float32)
    for i in range(sizey):
        for j in range(sizex):
            angle = polar[i][j][0]
            mag   = polar[i][j][1]
            
            color1 = (  (angle+m.pi)/(2*m.pi))*(mag/maxMag)*255
            color2 = (1-(angle+m.pi)/(2*m.pi))*(mag/maxMag)*255
            anglePic[i][j] = [color1,color2,0]

    
    arrows = np.zeros((sizey, sizex ,3),np.float32)
    for i in range(sizey):
        for j in range(sizex):
            ang,mag,u,v = polar[i][j]
            prt1 =int(j+u * arrowScale * mag/maxMag)
            prt2 =int(i+v * arrowScale * mag/maxMag)
            try:
                if i%arrowDensity == 0 and j%arrowDensity ==0:
                    cv2.arrowedLine(arrows,(j,i),(prt1,prt2),(255,0,0))

            except Exception as e:
                print(e)
                print(ang,mag,u,v)
                pass

                
    #cv2.imshow("IM1",image1)
    #cv2.imshow("IM2",image2)
    cv2.imwrite("1Color_"+file1, anglePic)
    cv2.imwrite("1Arrow_"+file1, arrows)
    print(file1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
   
if __name__ == "__main__":
    for i in range(6):
        main(i)
