import operator
import math
import random
from sympy import sympify
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from matplotlib import pyplot as plt

def loadnums(filename):
    """ Load x and y vectors from file """
    x_vec = []
    y_vec = []
    with open(filename) as myfile:
        lines = myfile.readlines()
        for index,line in enumerate(lines):
            line = line.strip().split()
            #data starts on 3rd line
            if index >= 2:
                x_vec.append(float(line[0]))
                y_vec.append(float(line[1]))
    print("X vector", x_vec, "size", len(x_vec))
    print("Y vector", y_vec, "size", len(y_vec))
    return x_vec,y_vec

def evalSymbReg(individual,x_points,y_points):
    """evaluate the fitness of the individual"""
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression and
    # the real values (y points)
    sqerrors = ((func(x_points[i]) - y_points[i])**2 for i in range(len(x_points)))
    return math.fsum(sqerrors) / len(x_points),

def create_primset():
    #create primitive set
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    #pset.addPrimitive(operator.mul, 2)
    #pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(math.cos,1)
    pset.addPrimitive(math.sin,1)
    pset.addEphemeralConstant("rand101", lambda: random.randint(-2,2))
    #use x instead of ARG0 as our variable name
    pset.renameArguments(ARG0='x')
    return pset    

def create_stats():
    #create tools for computing statistics on populations
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def human_readable(individual):
    """ convert a candidate function into a human readable string""" 
    #dictionary to translate deap output to human readable
    #using sympy
    locals = {
        'sub': lambda x, y : x - y,
        'protectedDiv': lambda x, y : x/y if y!=0 else 1,
        'mul': lambda x, y : x*y,
        'add': lambda x, y : x + y,
        'pow': lambda x, y : x**y,
        'neg': lambda x    : -x,
    }

    print(f'original: {individual}')
    expr = sympify(str(individual) , locals=locals)
    print(f'simplified: {expr}')
    return expr

def create_toolbox(pset,tree_depth):
    """given a pset, create a toolbox to operate while evolving"""
    #define evolution parameters with toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    #evaluate performance
    toolbox.register("evaluate", evalSymbReg, x_points=x_vec, y_points=y_vec)
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=tree_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=tree_depth))
    return toolbox

def plot_comparison(x_vec,y_vec,func):
    """compare a function's prediction with the real points"""
    x_vec = np.asarray(x_vec)
    pred = [func(i) for i in x_vec]
    plt.plot(x_vec,y_vec,marker="*",linestyle="None",label="realdata")
    plt.plot(x_vec,pred,label="prediction")
    plt.title(f"function ${best_str}$")
    plt.legend()
    plt.savefig("testimg.png")
    print("Plot saved.")    

if __name__=="__main__":
    filename = "regression.txt"
    generation_pop = int(input("Enter number of individuals per generation: "))
    gen_num = int(input("Enter number of generations: "))
    depth = int(input("Enter max depth of syntax tree (controls complexity of functions): "))
    r = int(input("Enter a random seed int: "))

    #load our x and y points
    x_vec, y_vec = loadnums(filename)
    print("X vector", x_vec, "size", len(x_vec))
    print("Y vector", y_vec, "size", len(y_vec))
    plt.plot(x_vec, y_vec, marker="*")
    plt.title("Raw data")
    plt.savefig("raw_data.png")
    #create primitive set of operations for tree generation
    pset = create_primset()

    #define fitness function and create individual who holds the genotype
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    #individual is a tree
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    #define evolution parameters with toolbox
    toolbox = create_toolbox(pset,depth) 

    #------------------------------------------
    #run it
    random.seed(r)

    pop = toolbox.population(n=generation_pop)
    hof = tools.HallOfFame(1)

    #grab statistics
    mstats = create_stats()


    #eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,halloffame=None, verbose=__debug__):
    """This algorithm implements a simple evolutionary algorithm
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution operators.
    :param cxpb: The probability of mating two individuals. Set to 50%
    :param mutpb: The probability of mutating an individual. Set to 30%
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object 
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: population: The final population
    :returns: run_log: statistics of the evolution process
    """
    population, run_log = algorithms.eaSimple(pop, toolbox, 0.5, 0.3, gen_num, stats=mstats,
                                   halloffame=hof, verbose=True)
    print(run_log)

    #try to get best individual into a printable form
    best_ind = tools.selBest(population, 1)[0]
    best_str = human_readable(best_ind)
    print("best str", best_str)

    #now plot the function vs the data
    func = toolbox.compile(expr=best_ind)
    #vectorize so we can pass numpy array
    #func2 = np.vectorize(func)
    #save plot
    plot_comparison(x_vec,y_vec,func)
