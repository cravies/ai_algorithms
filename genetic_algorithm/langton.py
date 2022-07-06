import numpy as np
from matplotlib import pyplot as plt

class ant():
    def __init__(self, ant_row, ant_column, orientation):
        #matrix indexing: m,n
        self.row = ant_row
        self.column = ant_column
        #orientation is an array of [U,R,D,L]
        orient_dict = {"U":0, "R":1, "D":2, "L":3}
        self.orientation = orient_dict[orientation]

    def turn_left(self):
        #turn the ant left
        self.orientation = (self.orientation - 1) % 4

    def turn_right(self):
        #turn the ant right
        self.orientation = (self.orientation + 1) % 4

    def orient_get(self):
        reverse_dict = {0:"U",1:"R",2:"D",3:"L"}
        #get orientation
        return reverse_dict[self.orientation]

def move(grid,ant):
    cur_square = grid[ant.row,ant.column]
    
    if cur_square == 0:
        #on a black square
        ant.turn_right()
    elif cur_square == 1:
        #on a white square
        ant.turn_left()
    else:
        print("ERROR: cur square:",cur_square)

    #change the colour of the square the ant was on
    if grid[ant.row, ant.column]==1:
        grid[ant.row, ant.column]=0
    else:
        grid[ant.row, ant.column]=1

    #now we move based on orientation
    o = ant.orient_get()
    #periodic boundary conditions
    gridlen = np.shape(grid)[0]
    if o == "U":
        ant.row = (ant.row - 1) % gridlen
    elif o == "D":
        ant.row = (ant.row + 1) % gridlen
    elif o == "R":
        ant.column = (ant.row + 1) % gridlen
    elif o == "L":
        ant.column = (ant.row - 1) % gridlen
    else:
        print("ERROR: orientation ", o)

    return grid, ant

def plot_langton(grid,ant,i):
    #plot an ant on a grid
    plt.imshow(grid)
    plt.title("langton's ant, iter {}".format(i))
    plt.show()

def main():
    sz = 50
    grid = np.zeros([sz,sz])
    start = int(sz / 2)
    A = ant(start,start,"U")

    for i in range(1000):
        grid, A = move(grid,A)
        if (i%10==0):
            plot_langton(grid,A,i)

    plot_langton(grid,A)

if __name__=="__main__":
    main()
