# sudoku_solver

sudoku solver meant to solve sudoku puzzles. I was looking for a way to do a few puzzles without, you know, solving puzzles and came up with this simple script. when I got stuck I found this blog http://norvig.com/sudoku.html from which I borrowed heavily :)

usage:

    from sudoku_solver import sudokutable

    table = sudokutable()
    # a table is the 9 rows stuck one afteranother as a string. unknown numbers are a . 
   
    table.parse(".4.17...6......9..3..8..4152.4.38..9.........7..59.2.3418..6..7..3......5...81.3.")
    print(table)
    table.solve()
    print(table)
