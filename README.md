# ai_algorithms
Implementations of common AI algorithms from scratch.
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
