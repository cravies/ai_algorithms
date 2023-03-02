import random
filename = input("Enter filename with path: ")
out = input("Enter output filename with path: ")
lines = open(filename).readlines()
random.shuffle(lines)
open(out, 'w').writelines(lines)
