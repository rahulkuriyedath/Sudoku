from SudokuSolver import SudokuSolver

obj = SudokuSolver()
grid = obj.readSudoku()
obj.solve(grid)