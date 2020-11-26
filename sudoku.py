import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("sudoku2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Sharpen image
# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# image = cv2.filter2D(image, -1, kernel)

# Find contours, filter using contour approximation, aspect ratio, and contour area

h, w, d = image.shape
#print('h: '+str(h))

threshold_max_area = (h/9) * (w/9)*(1.2)
threshold_min_area = (h/9) * (w/9)*(0.8)

print(threshold_max_area, threshold_min_area)

mylist = np.zeros(shape=(9,9))
cwidth = w/9
cheight = h/9


cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
square_ct = 0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.035 * peri, True)
    x,y,w,h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(c) 
    if len(approx) == 4 and area < threshold_max_area and area > threshold_min_area and (aspect_ratio >= 0.9 and aspect_ratio <= 1.1):
        
        i = int(np.floor(x/cwidth))
        j = int(np.floor(y/cheight))
        mylist[j,i] = str(i)+str(j)
        
        
        print(pytesseract.image_to_string(thresh[y:h,x:w], config="--psm 13"))
        plt.imshow(thresh[y:h,x:w])
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        print(x,y,w,h)
        square_ct += 1
        

    #cv2.imshow("image", image)
# cv2.imshow("thresh", thresh)

if square_ct!= 9*9:
    print('Did not find correct number of boxes')
    print('Number of boxes: '+str(square_ct))
    
    
#plt.imshow(image)
