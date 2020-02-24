**Homework 3**

**Problem 1**

Implement the function ```get_random_data(w, b, mu, sigma, m)``` that generates random data for logisitic regression.
This function should return the array ```data``` of shape ```(m, 2)``` and the array ```labels``` of shape ```(m, 1)```.

The entries of the arrays should be generated as follows.  For each row ```i in {0, 1, ..., m-1}:

- Choose ```c=0``` with probability 1/2 and ```c=1``` with probability 1/2.  
- Choose ```x``` uniformly at random in the interval [0, 1). 
- Set ```y = w * x + b + (-1)^c * n``` where ```n```` is chosen according to the normal distribution with mean ```mu``` and standard deviation ```sigma```.
- The ith row of the array ```data``` consists of the entries ```x``` and ```y```.
- The ith entry of the vector ```labels``` is ```c```.

**Problem 2**

**Problem 3**
