from copy import copy, deepcopy


class sudokuCell:

    # pos is noted as "A2" = 1st row , 2nd value
    def __init__(self, pos ):
        self.pos = pos
        self.row = "ABCDEFGHI".index(pos[0])
        self.column = int(pos[1])-1
        self.values = {1,2,3,4,5,6,7,8,9}

        self.cluster = self.column//3 + 3*(self.row//3)

    def __str__(self):
        if len(self.values) > 1:
            value = "."
        else:
            value = str(list(self.values)[0])
        return value

    def detailed(self):
        result = "cell: {p} (row {r}, col {c}, clust {cluster}) values: {v}"
        return result.format(v = self.values, p = self.pos,
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

    def __init__(self, rows = "ABCDEFGHI", cols = "123456789"):
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

    def __iter__(self): return iter(self.cells)

    def __getitem__(self, pos):
        if type(pos) is str:
            return list(a for a in self if a.pos == pos)[0]
        elif type(pos) is int:
            return self.cells[pos]

    def __str__(self):
        '''returns a sudoku table in readable format'''
        result = ''
        i = 0
        for r in range(9):
            if r == 0:
                result+= '  012 345 678' + '\n'
            if r%3 == 0:
                result += '\n' # inserts the blank line after every 3 rows
            for c in range(9):
                if c == 0:
                    result += str(r)
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
        '''easy way to open a sudoku table. parse will read a string of sudoku
           values and set those values within an emply table. all non-digits and
           periods are ignored. Note that
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

        return cell_strike and set_strike

    def uniques(self):
        ''' loop through all unsolved cells untill either the sudoku has been
            solved or no new certainties can be found.
            the main loop of this function goes through all unsolved cells and
            tries to find a situation where a value appears only once.
            eg, take the following row
            {1}, {2,3}, {3,4,5}, {3,4,5}, {3,4,5}, {6}, {7}, {8}, {9}
            we know that the second cell can be either 2 or 3, but because 2
            does not appear as an option in the other cells, we know that it
            must appear in the second cell
            the loop exits when either all cells have been solved or the main
            loop has run an entire cycle without finding a new cell that can be
            solved in this way
        '''
        solved_new = True # solved_new is set to true so the main loop runs at least once
        while solved_new and not self.solved():
            solved_new = False
            # this is the main loop
            for cell in (c for c in self if len(c.values) > 1):
                for group in ['row', 'column', 'cluster']:
                    possibles = copy(cell.values)
                    # check each neighbouring row, column, cluster to see if
                    # cell has unique value do this by removing all values that
                    # also appear in neighbouring cells.
                    for c in self.get_set(cell, group):
                        possibles -= c.values
                    if len(possibles) == 1: # cell has one unique possible
#                        print (cell.detailed(), 'possibles', possibles)
                        # because set_value returns true on succes, solved_new
                        # is true and the loop can run again.
                        solved_new = self.set_value(cell, list(possibles)[0])
                        break # exit the loop and continue with the next cell
        return self.solved()


    def solve(self):
        '''solve calls uniques. ith  and then tries guessing
        '''
        if self.uniques():
            return True
        min_length = min (len(c.values) for c in self if len(c.values) > 1)
        for cell in (c for c in self if len(c.values) == min_length):
                for v in cell.values:
                    print('testing value {} for cell {}'.format(v, cell.pos))
                    copyself = deepcopy(self)
                    copyself.set_value(copyself[cell.pos], v)
                    copyself.uniques()
                    if copyself.solved():
                        self.set_value(cell, v)
                        self.uniques()
                        return True
                    else: copyself.seek()
        return False

    def solved(self):
        return all(len(c.values)==1 for c in self)

#    def solve(self):
#        i = 0
#        self.uniques()
#        i += 1
#        print('solved in {} moves'.format(i))


def tests():

    puzzle1 = ".4.17...6......9..3..8..4152.4.38..9.........7..59.2.3418..6..7..3......5...81.3."

    puzzle2 = "..1....9...6.95..4.8.....21.....4..2...518...5..7.....12.....7.3..67.2...7....5.."

    puzzle3 = ".....5..8...2..43...649...7.9..3.5.2.7.....8.1.3.8..9.8...437...25..1...7..9....."
    table1 = sudokutable()
    table2 = sudokutable()
    table3 = sudokutable()

    table1.parse(puzzle1)
    table2.parse(puzzle2)
    table3.parse(puzzle3)

    table1.solve()
    print('the solved table 1')
    print(table1)


    print('this is table 2')
    table2.solve()
    print(table2)

    print('this is table 3')
    table3.solve()
    print(table3)

#    for c in (ce for ce in table2 if len(ce.values) > 1):
#        print(c.detailed())

if __name__ == "__main__":
    tests()
