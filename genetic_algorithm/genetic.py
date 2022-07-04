"""
Use a genetic algorithm to find the
global optima of the rosenbrock function.
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import rand
from random import randint
import operator

#set constants for rosenbrock
a=1
b=100

#population parameters
pop_size = 40

x_min = -2
x_max = 2
y_min = -1
y_max = 3
#scale to x and y range of plot
x_diff = x_max - x_min
y_diff = y_max - y_min

#iteration parameters
mut_rate = int(pop_size * 0.2)
cross_rate = int(pop_size * 0.2)
select_rate = cross_rate * 2

def rosenbrock(x,y):
    #compute rosenbrock function
    return (a-x)**2 + b*(y-x**2)**2

def init_population(n):
    #initialize n candidates
    pop = rand(n,2)
    return pop

def inverse(val):
    #protected inverse
    if val == 0:
        return 1
    else:
        return 1/val

def mutate(pop_x,pop_y):
    #random mutation of n of the candidates
    rng = len(pop_x)
    for i in range(0,mut_rate):
        r = randint(0,rng-1)
        r_x,r_y = pop_x[r],pop_y[r]
        print("chose point ",r," cords are",r_x,r_y)
        if rand() > 0.5:
            #permute x
            print("permute x")
            r_x = x_diff * rand() + x_min 
        else:
            #permute y
            print("permute y")
            r_y = y_diff * rand() + y_min
        print("permuted to: ",r_x,r_y)
        pop_x[r] = r_x
        pop_y[r] = r_y
    return pop_x,pop_y
    
def crossover(pop_x,pop_y):
    rng = len(pop_x)
    for i in range(0,cross_rate):
        p1 = randint(0,rng-1)
        p2 = randint(0,rng-1)
        x1,y1 = pop_x[p1],pop_y[p1]
        x2,y2 = pop_x[p2],pop_y[p2]
        print("parent 1 is ",x1,y1)
        print("parent 2 is ",x2,y2)
        #two parents produce two children
        child_1 = [x1,y2]
        child_2 = [x2,y1]
        #add these children to the population
        pop_x.append(child_1[0])
        pop_y.append(child_1[1])
        pop_x.append(child_2[0])
        pop_y.append(child_2[1])
        print("child 1 is ",child_1)
        print("child 2 is ",child_2)
    print("pop_x was size ",rng," is now size ",len(pop_x))
    return pop_x, pop_y 

def selection(pop_x,pop_y):
    #truncation selection.
    #remove the worst n individuals
    score = {}
    for i in range(0,len(pop_x)):
        x = pop_x[i]
        y = pop_y[i]
        #minimizing so fitness value inverse
        fitness = inverse(rosenbrock(x,y))
        score[f"{i}"] = fitness 
        print(f"point {i} at {x},{y} has fitness {fitness}")

    #get rid of weakest individuals
    sorted_score = dict(sorted(
            score.items(),
            key=operator.itemgetter(1),
            ))
    print(sorted_score)

    #indexes to be eliminated
    perished_inds = list(sorted_score.keys())[0:select_rate]
    print(perished_inds)
    perished_inds=[int(i) for i in perished_inds]

    #remove from population perished individuals
    pop_x = np.delete(pop_x, perished_inds)
    pop_y = np.delete(pop_y, perished_inds)

    return list(pop_x), list(pop_y)

def iterate(pop_x,pop_y):
    #mutate a couple of points
    pop_x, pop_y = mutate(pop_x,pop_y)
    #crossover
    pop_x, pop_y = crossover(pop_x, pop_y)
    #selection
    pop_x, pop_y = selection(pop_x, pop_y)
        
    return pop_x, pop_y

def main():
    #plot rosenbrock
    x = np.linspace(x_min,x_max,1000)
    y = np.linspace(y_min,y_max,1000)
    xx, yy = np.meshgrid(x,y)
    zz = rosenbrock(xx,yy)
    plt.contour(x,y,zz,50)
    plt.title("Rosenbrock function.")

    #grab population of points and plot
    pop = init_population(pop_size)
    print("pop is", pop)
    pop_x = pop[:,0]
    pop_y = pop[:,1]
    #scale to range of plot
    pop_x = [x_diff * i + x_min for i in pop_x]
    pop_y = [y_diff * i + y_min for i in pop_y]
    print("x min", min(pop_x), "x_max", max(pop_x))
    print("y min", min(pop_y), "y_max", max(pop_y))
    plt.scatter(pop_x,pop_y,color="red")
    plt.show()

    for i in range(300):
        """
        iterate over generations
        """
        #plot rosenbrock
        x = np.linspace(x_min,x_max,100)
        y = np.linspace(y_min,y_max,100)
        xx, yy = np.meshgrid(x,y)
        zz = rosenbrock(xx,yy)
        plt.contourf(x,y,np.log(zz),30)
        plt.title("Rosenbrock function.")
       
        #make new generation
        pop_x, pop_y = iterate(pop_x,pop_y)
        print(len(pop_x),len(pop_y))
        plt.scatter(pop_x,pop_y,color="red",marker="x")
        plt.show()

if __name__=="__main__":
    main()
