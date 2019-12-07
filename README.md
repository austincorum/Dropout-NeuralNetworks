# Dropout: A Simple Way to Prevent Neural Networks from Overfitting
In this research project, I will focus on the effects of changing dropout rates on the MNIST dataset. My goal is to reproduce the figure below with the data used in the research paper. The purpose of this project is to learn how the machine learning figure was produced. Specifically, learning about the affects to classification error when changing/not changing the dropout probability.
![figure](https://github.com/akc257/Dropout-A-Simple-Way-to-Prevent-Neural-Networks-from-OverfittingNeural-Networks-from-Overfitting/blob/master/DropoutFigure.png)
Figure Referenced from: Srivastava, N.,  Hinton, G.,Krizhevsky, A., Krizhevsky, I., Salakhutdinov, R., Dropout: A Simple Way to Prevent Neural Networks from Overfitting, Figure 9

# How to reproduce analysis:
- Must have python 2.7.X installed
- Use command "make" in terminal
- Best to run in a UNIX/Linux Environment

## Software/Libraries used
I used TensorFlow to run dropout on the MNIST dataset, Matplotlib to assist in recreating the figure in the paper. I also used a built in Decimal library to calculate the different values of p, from 0.0 to 1.0. The library "csv" was imported for adding previously run data into a CSV file, to save time in computation of already calculated values of p. Numpy was imported to get the plotting to have the same step size on the x and y-axis. Lastly, I imported "os" so that I could get rid of an error due to using a CPU rather than a GPU.

## Purpose of Coding Project:
Exploring the effects of varying values of the tunable hyperparameter 'p' (the probability of retaining a unit in the network) and the number of hidden layers, 'n', that affect error rates. When the product of p and n is fixed, we can see that the magnitude of error for small values of p has reduced (fig. 9a) compared to keeping the number of hidden layers constant (fig. 9b).

## Problem Setting:
With limited training data, many complicated relationships between inputs/outputs will be a result of sampling noise. They will exist in the training set, but not in real test data even if it is drawn from the same distribution. This complication leads to overfitting, this is one of the algorithms to help prevent it from occurring. The input for this figure is a dataset of handwritten digits, and the output after adding dropout are different values that describe the outcome of applying the dropout method. All in all, less error is outcome after adding dropout.

## Data Sources:
A real world problem that this can apply to is google searching, someone may be searching for a movie title but they might only be looking for images because they are more visual learners. So dropping out the textual parts, or brief explanations will help you focus on the image features. The article states where they retrieve the data from (http://yann.lecun.com/exdb/mnist/). Each image is a 28x28 digit representation. The y labels seem to be the image data columns.

## Algorithm:
My goal in reproducing this figure is to test/train the data and calculate the classification error for each probability of p (probability of retaining a unit in the network). My goal is to get p to increase as the error goes down to show that my implementation is valid, and I will tune this hyper parameter to get the same outcome. I will do this by looping through all the training and testing data using a 784-2048-2048-2048-10 architecture and keep the n fixed then change pn to be fixed. I will then gather/write the data into a csv file. This csv file will then contain all the necessary data to output the figures. In this project, I will learn how the dropout rate can benefit the overall error in a neural network.

### Project Proposal:
[Click to View](https://github.com/akc257/Dropout-A-Simple-Way-to-Prevent-Neural-Networks-from-OverfittingNeural-Networks-from-Overfitting/blob/master/Project%202%20Week%201.pdf)
