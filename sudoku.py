import sudoku_solver
import sudoku_imgreader
import os


def solve_sudoku(file):
    sudoku_pic = sudoku_imgreader.sudoku_image(file)
    numbers = sudoku_pic.cells_to_numbers()
    puzzle = sudoku_solver.sudokutable()
    puzzle.parse(numbers)
    sudoku_pic.show_image()
    while not puzzle.solved():
        cell = puzzle.solve_next()
        sudoku_pic.add_cell_number(cell.id, str(cell.value()))
    sudoku_pic.show_image()


def main():
    path = '/home/marc/projects/sudoku_solver/sudoku_images/'
    files = [f for f in os.listdir(path) if f.endswith('.jpg') ]
    files =sorted(files)
    for f in files:
        file_str = path+f
        print('attempting to open', file_str)
        solve_sudoku(file_str)

if __name__ ==  '__main__':
    main()
