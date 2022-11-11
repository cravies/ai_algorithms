"""
Use a genetic algorithm to find the
global optima of the rosenbrock function.
"""
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import randn, rand
from random import randint
import operator
from copy import copy
from PIL import Image
import os

def rosenbrock(x: float,y: float) -> float:
    """
    compute rosenbrock function
    """
    return (a-x)**2 + b*(y-x**2)**2

def init_population(n: int) -> np.array:
    """
    initialize n candidates
    """
    pop = rand(n,2)
    return pop

def inverse(val: float) -> float:
    """
    protected inverse
    """
    if val == 0:
        return 1
    else:
        return 1/val

def mutate(pop_x: list[float],pop_y: list[float]) -> list[list[float]]:
    """
    randomly mutate n of the candidates.
    iterate mut_rate times:
        pick random person p_r
        add small random noise 
        n ~ N(0,0.1 * I)
        to co-ordinates
        modulus (x_max, y_max) to stay 
        in the function range
    return [pop_x,pop_y]
    """
    rng = len(pop_x)
    for i in range(0,mut_rate):
        r = randint(0,rng-1)
        r_x,r_y = pop_x[r],pop_y[r]
        r_x = (r_x + 0.1*randn()) % x_max
        r_y = (r_y + 0.1*randn()) % y_max
        print("permuted to: ",r_x,r_y)
        pop_x[r] = r_x
        pop_y[r] = r_y
    return pop_x,pop_y

def crossover(pop_x: list[float],pop_y: list[float]) -> list[list[float]]:
    """
    iterate cross_rate times:
        choose two random parents
        p1 = [x1,y1]
        p2 = [x2,y2]
        breed two children from them 
        by crossing co-ordinates
        c1 = [x1,y2]
        c2 = [x2,y1]
        add c1,c2 to population.
    return [pop_x, pop_y]
    """
    sz = len(pop_x)
    for i in range(0,cross_rate):
        p1 = randint(0,sz-1)
        p2 = randint(0,sz-1)
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
    print("pop_x was size ",sz," is now size ",len(pop_x))
    return pop_x, pop_y 


def selection(pop_x: list[float],pop_y: list[float]) -> list[list[float]]:
    """
    Truncation selection.
    Given a population specified by:
    pop_x = [x1,x2,...,xn]
    pop_y = [y1,y2,...,yn]
    where p3 = [x3,y3]
    remove the n individuals with lowest fitness.
    The fitness is defined as 1/f(x) 
    as we are trying to find minima of f(x)
    returns [list(pop_x), list(pop_y)]
    """
    score = {}
    for i in range(0,len(pop_x)):
        x = pop_x[i]
        y = pop_y[i]
        #minimizing so fitness value inverse
        fitness = inverse(rosenbrock(x,y))
        score[f"{i}"] = fitness 

    #get rid of weakest individuals
    sorted_score = dict(sorted(
            score.items(),
            key=operator.itemgetter(1),
            ))

    #indexes to be eliminated
    perished_inds = list(sorted_score.keys())[0:select_rate]
    print(perished_inds)
    perished_inds=[int(i) for i in perished_inds]

    #remove from population perished individuals
    pop_x = np.delete(pop_x, perished_inds)
    pop_y = np.delete(pop_y, perished_inds)

    return list(pop_x), list(pop_y)

def iterate(pop_x: list[float],pop_y: list[float]) -> list[list[float]]:
    """
    perform mutation, crossover, 
    and selection on a population.
    input:
    pop_x, pop_y
    returns [pop_x, pop_y]
    """
    #mutate a couple of points
    pop_x, pop_y = mutate(pop_x,pop_y)
    #crossover
    pop_x, pop_y = crossover(pop_x, pop_y)
    #selection
    pop_x, pop_y = selection(pop_x, pop_y)
    return pop_x, pop_y

def plot_background(x_min: int,x_max: int,y_min: int,y_max: int) -> None:
    """
    contour plot of background function 
    (rosenbrock) r(x,y) given x,y range
    """
    x = np.linspace(x_min,x_max,1000)
    y = np.linspace(y_min,y_max,1000)
    xx, yy = np.meshgrid(x,y)
    zz = rosenbrock(xx,yy)
    plt.contourf(x,y,zz,30,cmap='plasma_r')
    plt.title("Rosenbrock function.")

def converged(x_old: list[float],x_new: list[float],y_old: list[float],y_new: list[float],tol: float) -> bool:
    """
    given a previous set of x and y coords
    that define a population, and a current set,
    calculate if they are significantly different
    """
    #don't care about order because GP shuffles it.
    x_old = np.sort(x_old)
    y_old = np.sort(y_old)
    x_new = np.sort(x_new)
    y_new = np.sort(y_new)
    if (np.allclose(x_old,x_new,rtol=tol) and 
        np.allclose(y_old,y_new,rtol=tol)):
        return True
    return False

def getint(name: str) -> int:
    """
    given a filename input in 
    the format num.png, return the 
    number num
    """
    num, _ = name.split('.')
    return int(num)

def make_gif(fp_out: str) -> None:
    """
    Given a input and output path in the format:
    fp_in = "/path/to/image_*.png"
    fp_out = "/path/to/image.gif"
    Convert the *ordered* images in the input 
    to the gif in the output.
    """
    files = os.listdir('./')
    files = sorted([f for f in files if '.png' in f], key=getint)
    print(files)
    imgs = (Image.open(f) for f in files)
    img = next(imgs)  # extract first image from iterator
    img.save(fp=fp_out, format='GIF', append_images=imgs,
            save_all=True, duration=300, loop=0)

def main(pop_size: int, x_min: float, x_max: float, y_min: float, y_max: float, tol: float, iters: int) -> None:
    #grab population of points and plot
    pop = init_population(pop_size)
    pop_x = pop[:,0]
    pop_y = pop[:,1]
    #scale to range of plot
    pop_x = [x_diff * i + x_min for i in pop_x]
    pop_y = [y_diff * i + y_min for i in pop_y]

    #relative convergence tolerance
    tol=2e-1
    #iterate over generations
    for i in range(iters):
        print("~"*30)
        print(f"ITER {i}")
        #convergence check
        pop_x_old = copy(pop_x)
        pop_y_old = copy(pop_y)
        pop_x, pop_y = iterate(pop_x,pop_y)
        if converged(pop_x_old,pop_x,pop_y_old,pop_y,tol):
            print(f"converged at iter {i}")
            break
        print(len(pop_x),len(pop_y))
        #plot results
        plot_background(x_min,x_max,y_min,y_max)
        plt.scatter(pop_x,pop_y,color="red",marker="x",label="population")
        #plot mean
        mean_x = np.mean(pop_x)
        mean_y = np.mean(pop_y)
        mean_fitness = inverse(rosenbrock(mean_x,mean_y))
        plt.scatter(1,1,color='green',marker='o',label="global minima")
        plt.scatter(np.mean(pop_x),np.mean(pop_y),color='blue',marker='o',label="polulation mean")
        plt.title(f"Mean sol fitness {mean_fitness:1f} \n Mean coords [{mean_x:2f},{mean_y:2f}]")
        plt.legend()
        plt.savefig(f'./{i}.png')
        plt.clf()

    #make gif of results
    #filepath for output
    fp_out = "./res.gif"
    make_gif(fp_out)
    #remove pngs used to make gif
    os.system('rm *.png')

if __name__=="__main__":
    os.system('rm *.png')
    os.system('rm res.gif')
    #population parameters
    pop_size = 300
    scaling = 1.0
    x_min = scaling*(-2)
    x_max = scaling*(2)
    y_min = scaling*(-1)
    y_max = scaling*(3)
    #scale to x and y range of plot
    x_diff = x_max - x_min
    y_diff = y_max - y_min
    #iteration parameters
    mut_rate = int(pop_size * 0.2)
    cross_rate = int(pop_size * 0.2)
    select_rate = cross_rate * 2
    #constants for rosenbrock
    a=1
    b=100
    #constants for iteration
    tol=1e-2
    iters=100
    #run
    main(pop_size, x_min, x_max, y_min, y_max, tol, iters)