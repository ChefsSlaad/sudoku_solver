# sudoku solver

sudoku solver meant to solve sudoku puzzles. I was looking for a way to do a few puzzles without, you know, solving puzzles and came up with this script.

I have later added the option to use scanned or photographed images of a sudoku to solve the sudoku.

usage:

    # text only table
    from sudoku_solver import sudokutable
    
    table = sudokutable(".4.17...6......9..3..8..4152.4.38..9.........7..59.2.3418..6..7..3......5...81.3.")
    # a table is the 9 rows stuck one afteranother as a string. unknown numbers are a . 
    print(table)
    table.solve()
    print(table)


with images:
    
    from sudoku import solve_sudoku
    
    solve_sudoku('sudoku_images/sudoku000.jpg')


next step: 
    * create a simple web interface so that an image captured with a camera can be immediately solved and returned
    



credits:  http://norvig.com/sudoku.html 
          https://stackoverflow.com/questions/59182827/how-to-get-the-cells-of-a-sudoku-grid-with-opencv
          https://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square
          https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

