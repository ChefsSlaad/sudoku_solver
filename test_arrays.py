import numpy as np

grid_array = np.zeros(shape = (100,2), dtype = np.int32)

i = 0
for y in range(10):
    for x in range(10):
        grid_array[i] = [x,y]
        i +=1

grid_array = grid_array.reshape((10,10,2))

def pretty_print(ugly_array):

    ys,xs,d = ugly_array.shape
    for y in range(ys):
        print_str = ''
        for x in range(xs):
            item = ugly_array[y,x]
            print_str += '[{:>2},{:>2}] '.format(item[0],item[1])
        print(print_str)

pretty_print(grid_array)


print(grid_array[1:3,1:3,:])
