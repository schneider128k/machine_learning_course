# HW 1 (under construction)

Train 10 classifiers that perform binary classification:

- Is the input image the digit i?
- Is it a digit different from i?

Use logistic regression with 
- mean square error loss
- binary cross-entropy loss

- Implement mini-batch stochastic gradient descent using only numpy, that is, you are not allowed to use Keras except for loading the data set.
- Use Keras

Round the grey values of the images to 1 and 0 so you obtain a black and white image. Add as an additional feature the number of white regions. For instance, a typical 0 has 2 white regions and 8 has 3. Use the following neighborhoods for pixels:

```
pixel x,y is connected to

 o
o.o  
 o
 
ooo
o.o
ooo
```

 
 
