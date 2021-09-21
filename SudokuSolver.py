import cv2
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image
from torchvision import transforms

class SudokuSolver:
    """
    This class is capable of solving a sudoku puzzle given in the form of an image.
    readSudoku method performs OCR on the image and return a list of list representation of the numbers
    solve method solves the puzzle given a list of lists. 
    """

    def readSudoku(self, path):
        """Performs OCR on sudoku image and return results in list of lists representation

        Returns:
            [list]: A nested list representing the unsolved 9 by 9 sudoku grid
        """
        #### 1. PREPARE THE IMAGE
        img = cv2.imread(path)
        img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
        imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
        imgThreshold = preProcess(img)

        # #### 2. FIND ALL COUNTOURS
        imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
        imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
        #cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

        #### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
        biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
        print(biggest)
        if biggest.size != 0:

            model = MNIST_classifier()                   # create an instance of the model
            model.load_state_dict(torch.load('models/Digit_CNN.pt', map_location=torch.device('cpu')))
            model.eval()

            biggest = reorder(biggest)
            print(biggest)
            cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
            pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
            imgDetectedDigits = imgBlank.copy()
            imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
            boxes = splitBoxes(imgWarpColored)
            print(len(boxes))
            cv2.imshow('a',boxes[0])
            cv2.imshow('b',boxes[9])
            cv2.imshow('c',boxes[80])


        else:
            print("No Sudoku Found")
            return 0

        cv2.waitKey(0)
        
        
        
        
        ##########
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

        for cellno,c in enumerate(boxes):
            img = Image.fromarray()
            
            
            i = int(np.round(x/cwidth))
            j = int(np.round(y/cheight))
            grid[j][i] = str(i)+str(j)

            ink_percent = (np.sum(c == 255)/(np.sum(c == 0) + np.sum(c == 255)))*100

            if ink_percent>3.5:
                grid[j][i] = int(pytesseract.image_to_string(c, config="--psm 13")[0])
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

    def findNextCellToFill(self, grid, i, j):
        for x in range(i, 9):
            for y in range(j, 9):
                if grid[x][y] == 0:
                    return x, y
        for x in range(0, 9):
            for y in range(0, 9):
                if grid[x][y] == 0:
                    return x, y
        return -1, -1
    
    def isValid(self, grid, i, j, e):
        rowOk = all([e != grid[i][x] for x in range(9)])
        if rowOk:
            columnOk = all([e != grid[x][j] for x in range(9)])
            if columnOk:
                # finding the top left x,y co-ordinates of the section containing the i,j cell
                secTopX, secTopY = 3 * (i // 3), 3 * (j // 3)  # floored quotient should be used here.
                for x in range(secTopX, secTopX + 3):
                    for y in range(secTopY, secTopY + 3):
                        if grid[x][y] == e:
                            return False
                return True
        return False

    def solveSudoku(self, grid, i=0, j=0):
        i, j = self.findNextCellToFill(grid, i, j)
        if i == -1:
            return True
        for e in range(1, 10):
            if self.isValid(grid, i, j, e):
                grid[i][j] = e
                if self.solveSudoku(grid, i, j):
                    return True
                # Undo the current cell for backtracking
                grid[i][j] = 0
        return False

    def generateSolvedImage(self, unsolved, solved):
        blank = cv2.imread('images/blank.png')
        width = blank.shape[1]
        height = blank.shape[0]

        x = np.arange(width//18, width-(width//18), width//9)
        y = np.arange((height//9)-5, height - 5, (height//9)-5)

        positions = []
        for i in y:
            for j in x:
                positions.append((j,i))

        # Set up variables for painting numbers on blank image
        font = cv2.FONT_HERSHEY_PLAIN  
        fontScale = 4   
        thickness = 2  # Line thickness of 2 px 

        for i in range(0,9):
            for j in range(0,9):
                org = positions[(i*9)+j]
                if unsolved[i][j]!=0:
                    color = (0, 0, 0)
                else:
                    color = (71,99,255) 

                newimage = cv2.putText(blank, str(solved[i][j]), org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
                
        
        cv2.imwrite('solution.png',newimage)
        print('Successfully created image of solved sudoku!')
        plt.imshow(newimage)
        cv2.imshow('image',newimage)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

# solved =[[6, 7, 2, 8, 9, 4, 3, 5, 1],
# [8, 3, 1, 5, 2, 7, 4, 6, 9],
# [5 ,4 ,9 ,1 ,6 ,3 ,2 ,8, 7],
# [1 ,5 ,7 ,3 ,8 ,2 ,9 ,4, 6],
# [3 ,9, 6, 4, 7, 5, 1, 2, 8],
# [2 ,8 ,4 ,9 ,1 ,6 ,5 ,7 ,3],
# [9 ,2 ,8, 6, 5, 1, 7, 3, 4],
# [7 ,6 ,3 ,2 ,4 ,9 ,8 ,1, 5],
# [4 ,1, 5, 7, 3, 8, 6, 9, 2]]

# unsolved = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
# [8, 3, 1, 5, 2, 7, 4, 6, 9],
# [5 ,4 ,9 ,1 ,6 ,3 ,2 ,8, 7],
# [1 ,5 ,7 ,3 ,8 ,2 ,9 ,4, 6],
# [3 ,9, 6, 4, 7, 5, 1, 2, 8],
# [2 ,8 ,4 ,9 ,1 ,6 ,5 ,7 ,3],
# [9 ,2 ,8, 6, 5, 1, 7, 3, 4],
# [7 ,6 ,3 ,2 ,4 ,9 ,8 ,1, 5],
# [4 ,1, 5, 7, 3, 8, 6, 9, 2]]

# obj = SudokuSolver()
# obj.generateSolvedImage(unsolved, solved)
