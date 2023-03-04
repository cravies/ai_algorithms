# ai_algorithms
My implementations of common AI algorithms from scratch.
Includes:

## Neural Networks
* CNN for shape recognition
* MLP from scratch in numpy and C++
* Darknet19 for fruit classification
* CNN for MNIST rotation detection
* Perceptron from scratch in numpy and C++
* Variational Autoencoder (ELBO loss)
* Variational Autoencoder (MMD loss)

## Statistical Learning
* K means clustering
* K nearest neighbours
* Naive bayes 
* Naive babes variable elimination

## Evolutionary Computation
* Genetic programming - symbolic regression for denoising
* Genetic programming for function fitting
* Genetic algorithm to solve rosenbrock function
* Genetic algorithm to solve the knapsack problem
* Langton’s Ant simulation

## Decision Tree
* Decision tree algorithm for classification

## Details

I mostly used numpy. For the complicated neural network models I used pytorch.
I wrote these programs for various classes I am taking in compsci graduate school.

Jupyter notebooks should be self contained.

For .py scripts, to run the algorithms you need to move the corresponding files from the Datasets folder to your current directory

It is very clearn in the main loop in each program which file it expects.

i.e to run the genetic algorithm which expects regression.txt 

``` bash
cp ../Datasets/regression.txt ./ 
python3 genetic_algorithm.py
```
