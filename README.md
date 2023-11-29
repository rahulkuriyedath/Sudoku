# Sudoku

This project uses OpenCV CNNs to solve a Sudoku puzzle simply from an image. 

- OpenCV is used to detect the sudoku grid and identify each individual box.
- A CNN trained to recognize digits is responsible for recognizing what digit (or blank) is present in each box.
- Once all the digits are identified, a Sudoku solving algorithm solves the puzzle and displays the result
