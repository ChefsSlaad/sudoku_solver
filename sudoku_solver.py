from copy import copy, deepcopy

class sudokuCell:

    # pos is noted as "A2" = 1st row , 2nd value
    def __init__(self, pos):
        self.pos = pos
        self.id = id
        self.solved = False
        self.row = "ABCDEFGHI".index(pos[0])
        self.column = int(pos[1])-1
        self.values = {1,2,3,4,5,6,7,8,9}
        self.id = (self.row*9) + self.column
        self.cluster = self.column//3 + 3*(self.row//3)

    def __str__(self):
        if len(self.values) > 1:
            value = "."
        else:
            value = str(list(self.values)[0])
        return value

    def value(self):
        if len(self.values) == 1:
            return list(self.values)[0]

    def detailed(self):
        result = "cell: {p} (id {i}, row {r}, col {c}, clust {cluster}) solvede {s} values: {v}"
        return result.format(v = self.values, p = self.pos, i = self.id, s=self.solved,
                             r = self.row, c = self.column, cluster = self.cluster)

    def strike(self, p):
        """ strike checks if a value can leaglly be discarded from the list of possibles
            values in that cell.
        """

        if len(self.values) == 1 and p in self.values:
            return False
        else:
            self.values.discard(p)
        return len(self.values) > 0

class sudokutable:
    #   colums
    #     0 1 2 3 4 5 6 7 8
    #    0
    #    1  0     1     2
    #  r 2
    #  o 3
    #  w 4  3     4     5
    #  s 5
    #    6
    #    7  6     7     8
    #    8
    #
    #
    # set of 81 cells

    def __init__(self, table = None, rows = "ABCDEFGHI", cols = "123456789"):
        """
        Initiate the table. create a 9x9 grid and populate each cell
        with a set of {1-9}.
        add an id to each cell as well in the form of A1, C5, etc,
        cells are also part of a list whih can be searched
        """
        self.cells = []
        for r in rows:
            for c in cols:
                self.cells.append(sudokuCell(r+c))
        if not table == None:
            self.parse(table)

    def __iter__(self): return iter(self.cells)

    def __getitem__(self, pos):
        if type(pos) is str:
            return [a for a in self if a.pos == pos][0]
        elif type(pos) is int:
            return self.cells[pos]

    def __str__(self):
        '''returns a sudoku table in readable format'''
        result = ''
        i = 0
        for r in range(9):
            if r == 0:
                result+= '  123 456 789' + '\n'
            if r%3 == 0:
                result += '\n' # inserts the blank line after every 3 rows
            for c in range(9):
                if c == 0:
                    result += 'ABCDEFGHI'[r]
                if c%3 == 0:
                    result += ' ' #insert the blank column after every 3 columns
                result += self[i].__str__()
                i += 1
            result += '\n'
        return result

    def get_set(self, cell, group='all'):
        '''get_set returns all the neigbours of a cell but not the cell itself
           so get_set(cell, 'row') will return all neighbours in that cells row
           get_set(cell, column) and get_set(cell, cluster) will return
           neighbours in column and cluster respectively)
           if group is all, all unique neighbouring cells will be returned
           '''
        if group == 'row': result = set(a for a in self if a.row == cell.row)
        elif group == 'column': result = set(a for a in self if a.column == cell.column)
        elif group == 'cluster': result = set(a for a in self if a.cluster == cell.cluster)
        elif group == 'all':
            result = set()
            result.update(self.get_set(cell, 'row'),
                          self.get_set(cell, 'column'),
                          self.get_set(cell, 'cluster'))
        result.discard(cell)
        return result

    def parse(self, grid):
        '''Parse will read a string of sudoku values and set those values within
           an emply table. all non-digits and periods are ignored. Note that
                .4.17...6
                ......9..
                3..8..415
                2.4.38..9
                .........
                7..59.2.3
                418..6..7
                ..3......
                5...81.3.
            is equivalent to
            .4.17...6......9..3..8..4152.4.38..9.........7..59.2.3418..6..7..3......5...81.3.
            '''
        legalvals = '0123456789.'
        if len(grid) != 81:
            raise ValueError("the number of cells should be exactly 81")
        if not all(c in legalvals for c in grid):
            raise ValueError("the input may only contain digits between 1 and 9 or .")

        for i, c in enumerate(grid):
            if c in "123456789":
                self.set_value(self[i],int(c))

    def set_value(self, cell, value):
        '''set_value sets the value of a cell and strikes  that value from the
           possible values of all the neghbouring cells (in cluster, row and
           column. If a value cannot be set (because it contradicts an earlier
           value) it returns false. Otherwise it returns true
        '''
        remaining = {1,2,3,4,5,6,7,8,9} # use copy to keep the original list intact
        remaining.discard(value)
        set_strike = all( c.strike(value) for c in self.get_set(cell))
        cell_strike = all(cell.strike(v) for v in remaining)
        if set_strike and cell_strike:
            cell.solved = True

        return cell.solved

    def find_unique(self, cell):
        '''
        check if a cell is unsolved and try to find if a value only appears once
        eg, take the following row
        {1}, {2,3}, {3,4,5}, {3,4,5}, {3,4,5}, {6}, {7}, {8}, {9}
        we know that the second cell can be either 2 or 3, but because 2
        does not appear as an option in the other cells, we know that it
        must appear in the second cell
        '''
        if cell.solved:
            return False # allready unique, no new solution found

        elif len(cell.values) == 1:
            return self.set_value(cell, list(cell.values)[0])
        # check all cells that the cell forms a group with and remove the
        # possible values from the set possibles. If the only value left is
        # a unique value, set that as the new value for that cell
        for g in ('row', 'column', 'cluster'):
            possibles = copy(cell.values)
            for c in self.get_set(cell, group = g):
                possibles -= c.values
            if len(possibles) == 1:
                return self.set_value(cell, list(possibles)[0])
        return False

    def find_next_unique(self):
        '''
        loop through cells untill a new cell is found with unique value
        and return that cell
        '''
        for cell in self:
            if self.find_unique(cell):
                return cell
        return None

    def guess(self):
        '''
        find the cell with the minimum number of possible values and try
        each in turn. If the sudoku can be solved with that solution,
        enter it and return that cell.
        '''
        if self.solved(): return None
        unsolved_cells = sorted((c for c in self if len(c.values) > 1),
                            key=lambda  cell: len(cell.values))
#        print(list((c.pos, c.values) for c in unsolved_cells))
        for cell in unsolved_cells:
            for v in cell.values:
                copyself = deepcopy(self)
                copyself.set_value(copyself[cell.pos], v)
                copyself.solve(False)
                if copyself.solved():
                    self.set_value(cell, v)
                    return cell
        return None

    def solve_next(self, guess = True):
        '''
        search for a unique cell. if none can be found try to guess a value.
        loop through all unsolved cells untill and either the sudoku has been
        solved or no new certainties can be found.
        the loop exits when either all cells have been solved or the main
        loop has run an entire cycle without finding a new cell that can be
        solved in this way
        '''
        if not self.solved():
            unique = self.find_next_unique()
            if unique:
                return unique
            elif guess:
                return self.guess()
        return None

    def solve(self, guess = True):
        '''
        solve tries to solve the entire puzzle in one go. It exits if no new
        solution can be found (i.e. the puzzle cannot be solved)
        '''
        run = True
        while run and not self.solved():
            run = self.solve_next(guess)

    def solved(self):
        '''return True if all cells have been solved, False otherwise'''
        return all(c.solved for c in self)


def tests():
    puzzles = [".4.17...6......9..3..8..4152.4.38..9.........7..59.2.3418..6..7..3......5...81.3.",
               "..1....9...6.95..4.8.....21.....4..2...518...5..7.....12.....7.3..67.2...7....5..",
               ".....5..8...2..43...649...7.9..3.5.2.7.....8.1.3.8..9.8...437...25..1...7..9.....",
               ".9..8....24.3...1......53...2..7.1..4..1.6..2..1.3..9...56......6...4.87....2..5.",
               "42....7..6..1.........5.8.6.1.2.7.....2...9.....8.9.4.2.9.6.........1..2..1....58"
               ]
    for puzzle in puzzles:
        table = sudokutable()
        table.parse(puzzle)
        table.solve()
        print(table)

def test1():
#    puzzle = "..1....9...6.95..4.8.....21.....4..2...518...5..7.....12.....7.3..67.2...7....5.."
    puzzle = "42....7..6..1.........5.8.6.1.2.7.....2...9.....8.9.4.2.9.6.........1..2..1....58"

    table = sudokutable()
    table.parse(puzzle)
    while not table.solved():
        cell = table.solve_next()
#        input('found {} with value {}'.format(cell.pos, list(cell.values)[0]))

    print(table)


if __name__ == "__main__":
    tests()
#    test1()
