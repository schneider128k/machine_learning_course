
# Keras basics

## Layers: the building blocks of deep learning

- The fundamental data structure in neural networks is the **layer**.
- A layer is a data-processing module that takes as input one or more tensors and that outputs one or more tensors.
- Some layers are stateless, but more frequently layers have a state: the layers **weights**, one or several tensors learned with stochastic gradient descent, which together contain the network's **knowledge**.

## Layers: the building blocks of deep learning

- Different layers are appropriate for different tensor formats and different types of data processing.

- Simple vector data, stored in 2D tensors of shape ```(samples, features)``` is often processed by **densely connected layers**, 
also called **fully connected layers** (the ```Dense``` class in Keras).

- Sequence data, stored in 3D tensors of shape ```(samples, timesteps, features)```, is typically processed by **recurrent** layers 
such as **long-short term memory (LSTM)** layer.

- Image data, stored in 4D tensors, is usually processed by 2D **convolutional** layers ```Con2D```. 

Links to Keras documentation:

- [Core layers](https://keras.io/layers/core/)
- [Convolutional layers](https://keras.io/layers/convolutional/)
- [Recurrent layers](https://keras.io/layers/recurrent/)

