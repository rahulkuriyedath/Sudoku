import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract

class SudokuSolver:

    def readSudoku(self):
        image = cv2.imread("imagefinal.png")

        # If image smaller than square of 1260 size, resize to 1260.
        # This is what is working best for OCR
        if image.shape[0]<1260 or image.shape[1]<1260:
            image = cv2.resize(image, (1260,1260))
            
        # Convert to grayscale and binary threshold the image    
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        thresh_norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Find contours, filter using contour approximation, aspect ratio, and contour area
        h, w, d = image.shape

        # Only look at contours with area approximately = the area of individual square
        threshold_max_area = (h/9) * (w/9)*(1.2)
        threshold_min_area = (h/9) * (w/9)*(0.8)

        #grid = np.zeros(shape=(9,9))
        grid = [
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0]
        ]

        cwidth = w/9
        cheight = h/9

        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        square_ct = 0
        for cellno,c in enumerate(cnts):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.035 * peri, True)
            x,y,w,h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(c) 
            if len(approx) == 4 and area < threshold_max_area and area > threshold_min_area and (aspect_ratio >= 0.9 and aspect_ratio <= 1.1):
                i = int(np.round(x/cwidth))
                j = int(np.round(y/cheight))
                grid[j][i] = str(i)+str(j)

                ink_percent = (np.sum(thresh[y:y+h,x:x+w] == 255)/(np.sum(thresh[y:y+h,x:x+w] == 0) + np.sum(thresh[y:y+h,x:x+w] == 255)))*100
                
                if ink_percent>3.5: #3.5
                    #print(f"{x}, {y}, {w}, {h}")
                    #print(f"\nAt cell {j}, {i}")
                    #print('Found number: '+pytesseract.image_to_string(image[y:y+h,x:x+w], config="--psm 13"))
                    #gray[y:y+h,x:x+w]
                    grid[j][i] = int(pytesseract.image_to_string(image[y:y+h,x:x+w], config="--psm 13")[0])
                else:
                    grid[j][i] = 0
                    
                cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                #print(f"{x}, {y}, {w}, {h}")
                square_ct += 1
                print(f"Analysing cell {square_ct}/81")

        if square_ct!= 9*9:
            print('Did not find correct number of boxes')
            print('Number of boxes: '+str(square_ct))
            plt.imshow(image)
            return 0
        else:
            return grid

    def possible(self, x, y, n, grid):
        for i in range(0, 9):
            if grid[i][x] == n and i != y: # Checks for number (n) in X columns
                return False


        for i in range(0, 9):
            if grid[y][i] == n and i != x: # Checks for number (n) in X columns
                return False

        x0 = (x // 3) * 3
        y0 = (y // 3) * 3
        for X in range(x0, x0 + 3):
            for Y in range(y0, y0 + 3):  # Checks for numbers in box(no matter the position, it finds the corner)
                if grid[Y][X] == n:
                    return False    
        return True

    def solve(self, grid):
        for y in range(9):
            for x in range(9):
                if grid[y][x] == 0:
                    for n in range(1, 10):
                        if self.possible(x, y, n, grid):
                            grid[y][x] = n
                            self.solve(grid)
                            grid[y][x] = 0
                    return
        print('Sudoku solved:-')
        print(np.matrix(grid))