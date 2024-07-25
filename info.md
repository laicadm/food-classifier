# Food Classification
The objective of this project is to create a program that will be able to classify food images with the use of computer vision technology.
Keras will be used for the experimentation of neural networks to build and train models. 
Regarding the dataset, food images were acquired from the Food101 dataset, and the others are personally acquired and compiled by the contributors of this project.  

## Process
### A. Defining a model
Keras provides various types of models but in this project, the scope will be on Sequential model.
It is primarily used when the model consists of a single stack of layers, and each layer has exactly one input and output tensor.

### B. Adding layers to the model
Keras also provides various layers, which are designed to transform input data.
The scope will be the following: core layers (e.g. Flatten, Dense); convolutional layers (e.g. Conv2); pooling layers (e.g. MaxPooling2D)

#### Flattening
To grasp this concept, think of an image as a piece of paper divided into a grid of squares. For a 28x28 image, this means the paper has 28 rows and 28 columns, making 784 squares in total. Flattening the grid means taking all the squares and laying them out in a single line. Imagine cutting the paper into strips (rows) and then putting all those strips end-to-end to make one long strip. Now, instead of a 2D grid (28 rows and 28 columns), you have a 1D line with 784 numbers in a row. When you feed data into certain types of neural network layers (like Dense layers), they expect the input to be a long list of numbers (a 1D array) rather than a 2D grid. Flattening converts your 2D image into this expected format.

### C. Compilation of the model
Compiling a model in Keras means specifying how the model will learn and evaluate itself. This involves choosing an optimizer, a loss function, and metrics to monitor. 
1. The optimizer is an algorithm that adjusts the weights of the neural network to minimize the loss function. 
2. The loss function measures how well the model's predictions match the true data. The optimizer tries to minimize this value during training.
3. Metrics are used to evaluate the performance of the model. They are not used to train the model but to monitor and report the training progress.

