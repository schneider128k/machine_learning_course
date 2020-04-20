### Homework 4

**Problem 1**

Using only ```numpy```, implement the function ```conv2d```.  It takes as input ```input_mat``` and ```kernel_mat``` and outputs ```output_mat```.  All variables 
are square matrices.  It should compute the convolution of ```input_mat``` with the kernel ```kernel_mat``` using valid padding.

Note that the size ```s``` of the kernel matrix can also be an even number.  

```
input matrix

 0  1  2  3  4 

 5  6  7  8  9
 
 0  1  2  3  4 
 
 5  6  7  8  9
 
 0  1  2  3  4 

a) indicates the very first position in which a 2 x 2 filter is placed in the above 5 x 5 input matrix

+----+
|0  1| 2  3  4 
|    |
|5  6| 7  8  9
+----+
 0  1  2  3  4 

 5  6  7  8  9
 
 0  1  2  3  4 

b) indicates the very first position in which a 3 x 3 filter is placed in the above 5 x 5 input matrix

+-------+
|0  1  2| 3  4 
|       |
|5  6  7| 8  9
|       |
|0  1  2| 3  4 
+-------+
 5  6  7  8  9
 
 0  1  2  3  4 
```


**Problem 2**

Using only ```numpy```, implement the function ```maxpooling2d```. It takes as input ```input_mat``` and ```s``` and outputs ```output_mat```.
The variables ```input_mat``` and ```output_mat``` are square matrices and ```s``` is an integer.  It should compute the maxpooling operation 
on ```input_mat``` using window of shape ```s``` times ```s```.

---

Make sure that you throw appropriate custom exceptions indicating the problem when the operations in Problem 1 and Problem 2 cannot be performed. 

The test cases are in [test_cases_for_problems_1_2_homework_4.ipynb](https://colab.research.google.com/drive/1xuzLt4vwvt2Qt11vdKEOHrDyPX1IxnpO).

---

**Problem 3**

You will adapt the notebook [using VGG16 conv base for feature extraction, using data augmentation, not using dropout, fine-tuning](https://colab.research.google.com/drive/1F-RWvoxH8MmT7c1UmNy41iuOp-ejiLoF).
You will have to replace the VGG16 conv base by new conv bases. You should not use VGG19.

You should create two notebooks.  Both should use the same conv base, unfreeze the same number of layers of the conv_base, 
but use different classifiers.
