---
layout:     post
title:      Cheat sheet
date:       2017-12-17 17:53:00
summary:    Everything you should know about ML
categories: ML
thumbnail: connectdevelop
tags:
 - ML
 - optimization
 - Deep
 - Learning
 - CNN
---

# Basics in ML

## Famous Networks

### [Neocognitron](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.569.5982&rep=rep1&type=pdf) 
![](https://i.imgur.com/fCzuGR0.png)

A hierarchical multi-layered neural network, proposed by Kunihiko Fukushima in 1982.
It has been used for handwritten character recognition and other pattern recognition tasks.
Since backpropagation had not yet been applied for training neural nets at the time, it was limited by the lack of a training algorithm.<br />
[source](https://ml4a.github.io/ml4a/convnets/)

<br>

### [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) 
![](https://i.imgur.com/hmhIwri.png)


A pioneering digit classification neural network by LeCun et. al.
It was applied by several banks to recognise hand-written numbers on checks.
The network was composed of three types layers: convolution, pooling and non-linearity, and was trained using the backpropagation algorithm.<br />
[source](https://en.wikipedia.org/wiki/Convolutional_neural_network)

[![LeNet demo from 1993](https://img.youtube.com/vi/FwFduRA_L6Q/default.jpg)](https://www.youtube.com/watch?v=FwFduRA_L6Q "LeNet")

### [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 
![](https://i.imgur.com/vZxIoKA.png)


A convolutional neural network, which competed in the ImageNet Large Scale Visual Recognition Challenge in 2012 and achieved a top-5 error of 15.3%, more than 10.8% ahead of the runner up.
AlexNet was designed by Alex Krizhevsky, Geoffrey Hinton, and Ilya Sutskever.
The network consisted of five convolutional layers, some of which were followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax.
All the convolutional and fully connected layers were followed by the ReLU nonlinearity.<br />
[source](https://en.wikipedia.org/wiki/AlexNet)
### [VGGNet](https://arxiv.org/pdf/1409.1556.pdf)

![](https://i.imgur.com/Cz172Ar.png)

A 19 layer convolutional neural network from VGG group, Oxford, that was simpler and deeper than AlexNet.
All large-sized filters in AlexNet were replaced by cascades of 3x3 filters (with nonlinearity in between).
Max pooling was placed after two or three convolutions and after each pooling the number of filters was always doubled.
[source](https://www.cs.toronto.edu/~frossard/post/vgg16/)

### [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
![](https://i.imgur.com/WEU203j.png)


Developed by Microsoft Research, ResNet won first place in ILSVRC 2015 image classification using a 152-layer network -- 8 times deeper than the VGG.
The basic element in this architecture is the residual block, which	contains two paths between the input and the output, one of them being direct.
This forces the network to learn the features on top of already available input, and facilitates the optimization process.

---

## Networks Architectures

### Convolutional neural network (CNN)

![](https://i.imgur.com/iGDvrc2.png)


A multi-layer neural network constructed from convolutional layers. These apply a convolution operation to the input, passing the result to the next layer. The weights in the convolutional layers are shared, which means that the same filter bank is used in all the spatial locations.
[source](https://en.wikipedia.org/wiki/Convolutional_neural_network)

### Deconvolutional networks

![](https://i.imgur.com/FEZXF7q.png)


A generative network that is a special kind of convolutional network that uses transpose convolutions, also known as a deconvolutional layers. 

### [Generative Adversarial Networks (GAN)](https://arxiv.org/pdf/1406.2661.pdf)

![](https://i.imgur.com/vlcVHGM.png)


A system of two neural networks, introduced by Ian Goodfellow et al. in 2014, contesting with each other in a zero-sum game framework. The first is a deconvolutional network, G, that generates signals. While the second is a classifier, D, that learns to discriminates between signals from the true data distribution and fake ones produced by the generator. The generative network's goal is to increase the error rate of the discriminative network by fooling it with synthesized examples that appear to have come from the true data distribution.
[source](https://en.wikipedia.org/wiki/Generative_adversarial_network)



### Recurrent neural networks (RNN)

![](https://i.imgur.com/QUHiaFY.png)


RNNs are built on the same computational unit as the feed forward neural network, but differ in the way these are connected. Feed forward neural networks are organized in layers, where information flows in one direction -- from input units to output units -- and no cycles are allowed. RNNs, on the other hand, do not have to be organized in layers and directed cycles are allowed. This allows them to have internal memory and as a result to process sequential data. One can convert an RNN into a regular feed forward neural network by "unfolding" it in time, as depicted in the figures.

---

## Architectural Components/Motifs

### Definitions

#### Depth of a network

The number of layers in the network.

#### Feature vector / representation / volume

![](https://i.imgur.com/dw8KynP.png)


A three dimensional tensor of size W \times H \times D obtained in a certain layer of a neural network. W is the width, H is the height and D is the depth, i.e., the number of channels. If there is more than one example, this becomes a four dimensional tensor of size W \times H \times D \times B , where B is the batch size. image source

#### Spatial invariant feature vector

A feature vector that remains unchanged even if the input to the network is spatially translated.

#### Filters and biases

![](https://i.imgur.com/VNH6pkj.png)


Filters are a four dimensional tensor of size  F \times F \times D \times K and biases are a vector of length  K .  F is the width and height of the filter,  D is the number of channels and  K is the number of filters.

 

#### Neighbourhood

![](https://i.imgur.com/eVyMmFH.png)


A group of consecutive entries in a two-dimensional signal that has a rectangular or a square shape. image source

### Weight Layers

#### Convolutional (Conv) layer

![](https://i.imgur.com/2rfaIPz.png)


Accepts as input:
- feature vector of size $W_1 \times H_1 \times D_1$
- filters of size  $F \times F \times D_1 \times D_2$
- biases of length  $D_2$
- stride  $S$
- amount of zero padding  $P$

Outputs another feature vector of size  $W_2 \times H_2 \times D_2$ , where
- $W_2 = \frac{W_1-F+2P}{S}+1$
- $H_2 = \frac{H_1-F+2P}{S}+1$

The d-th channel in the output feature vector is obtained by performing a valid convolution with stride  S of the d-th filter and the padded input.
[source](http://cs231n.github.io/convolutional-networks/)

#### Stride

![](https://i.imgur.com/Zy7rKaS.png)


The amount by which a filter shifts spatially when convolving it with a feature vector.
[source](http://cs231n.github.io/convolutional-networks/)


#### Dilation

![](https://i.imgur.com/36zPXcY.png)


A filter is dilated by a factor  $Q$ by inserting in every one of its channels independently  $Q-1$ zeros between the filter elements.
[source](http://cs231n.github.io/convolutional-networks/) 


#### Fully connected (FC) layer

![](https://i.imgur.com/hzYyV0W.png)


In practice, FC layers are implemented using a convolutional layer. To see how this might be possible, note that when an input feature vector of size $H \times W \times D_1$ is convolved with a filter bank of size  $H \times W \times D_1 \times D_2$ , it results in an output feature vector of size  $1 \times 1 \times D_2$ . Since the convolution is valid and the filter can not move spatially, the operation is equivalent to a fully connected one. More over, when this feature vector of size $1 \times 1 \times D_2$ is convolved with another filter bank of size  $1 \times 1 \times D_2 \times D_3$ , the result is of size  $1 \times 1 \times D_3$ . In this case, again, the convolution is done over a single spatial location and therefore equivalent to a fully connected layer.
[source](http://cs231n.github.io/linear-classify/) 

#### Linear classifier

![](https://i.imgur.com/Fin8XWu.png)

This is implemented in practice by employing a fully connected layer of size  H \times W \times D \times C , where  C is the number of classes. Each one of the filters of size  H \times W \times D corresponds to a certain class and there are  C classifiers, one for each class. 

### Pooling Layers

#### Pooling Layer

![](https://i.imgur.com/awYfSFk.png)


Accepts as input:

- feature vector of size  $W_1 \times H_1 \times D_1$
- size of neighbourhood  $F$
- stride $S$

Outputs another feature vector of size W_2 \times H_2 \times D_1 , where Accepts as input:
- $W_2 = \frac{W_1 - F}{S} + 1$
- $H_2 = \frac{H_1 - F}{S} + 1$

The pooling resizes independently every channel of the input feature vector by applying a certain function on neighbourhoods of size  $F \times F$ , with a stride  $S$ .
[source](http://cs231n.github.io/convolutional-networks/)

#### Max pooling

Picks the maximal value from every neighbourhood.

#### Average pooling

Computes the average of every neighbourhood.

### Other Layers

#### Batch normalization

Accepts as input:

- feature vector of size  $W \times H \times D$
- bias vector of size  $D$
- gain vector of size  $D$

Outputs another feature vector of the same size. This layer operates on each channel of the feature vector independently. First, each channel is normalized to have a zero mean, unit variance and then it is multiplied by a gain and shifted by a bias. The purpose of this layer is to ease the optimization process.

#### Softmax layer

Takes the output of the classifier, applies exponent on the score assigned to each class and then normalizes the result to unit sum. The result can be interpreted as a vector of probabilities for the different classes.

### Nonlinearities

#### Sigmoid

![](https://i.imgur.com/PXmUHMm.png)


The sigmoid, defined as  $f(x) = \frac{1}{1 + e^{-x}}$ , is a non-linear function that suffers from saturation.

#### Saturation of activation

An activation that has an almost zero gradient at certain regions. This is an undesirable property since it results in slow learning. 


#### Tanh

![](https://i.imgur.com/Y9plGkf.png)


This non-linearity squashes a real-valued number to the range  $[-1, 1]$ . Like the sigmoid neuron, its activations saturate, but unlike the sigmoid neuron its output is zero-centered. 

#### ReLu

![](https://i.imgur.com/AeKhUFV.png)


The most popular non-linearity in modern deep learning, partly due to its non-saturating nature, defined as  $f(x) = \max(x,0)$ . 

#### Dead filter

A filter which always results in negative values that are mapped by ReLU to zero, no matter what the input is. This causes backpropagation to never update the filter and eventually, due to weight decay, it becomes zero and "dies".

#### Leaky ReLu

![](https://i.imgur.com/O1NjGmX.png)


A possible fix to the dead filter problem is to define ReLU with a small slope in the negative part, i.e.,  $f(x) = \left\{\begin{array}{lr} ax, & \text{for } x<0\\ x, & x \geq 0 \end{array}\right\}$ . 
image source

---

## Regularization in Neural Networks

### Dropout

![](https://i.imgur.com/G1IPNm6.png)


Accepts as input:

- feature vector of size  $H \times W \times D$
- probability  $p$

Outputs another feature vector of the same size. At train time, every neuron in it is set to the value of the corresponding neuron in the input with probability  $p$, and zero otherwise. At test time, the output feature vector is equal to the input one scaled by  $p$.

### Weight decay

Soft  $L_2$ constraint on the parameters of the network. This is done by decreasing every parameter in each iteration of SGD by its value times a small constant, corresponding to the strength of the regularization.

### Max norm constraints

Hard  $L_2$ constraint on the parameters of the network. This is done by imposing an upper bound on the  $L_2$ norm of every filter and using projected gradient descent to enforce the constraint.
source

### Data augmentation

Creating additional training samples by perturbing existing ones. In image classification this includes randomly flipping the input, cropping subsets from it, etc.

---

## Learning Ideas

### Gradient descent

![](https://i.imgur.com/bKW3li0.png)


To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient of the function at the current point.
[source](https://en.wikipedia.org/wiki/Gradient_descent)


### Stochastic gradient descent (SGD)

A stochastic approximation of the gradient descent for minimizing an objective function that is a sum of functions. The true gradient is approximated by the gradient of a randomly chosen single function.
[source](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

### Initialization of a network

Usually, the biases of a neural network are set to zero, while the weights are initialized with independent and identically distributed zero-mean Gaussian noise. The variance of the noise is chosen in such a way that the magnitudes of input signals does not change drastically.
[source](https://arxiv.org/pdf/1502.01852.pdf)

### Learning rate

The scalar by which the negative of the gradient is multiplied in gradient descent.

### Backpropagation

![](https://i.imgur.com/A7oFg65.png)


An algorithm, relying on an iterative application of the chain rule, for computing efficiently the derivative of a neural network with respect to all of its parameters and feature vectors.
[source](https://en.wikipedia.org/wiki/Backpropagation) 


### Goal function

The function being minimized in an optimization process, such as SGD.

### Data preprocessing

![](https://i.imgur.com/03vW7X5.png)


The input to a neural network is often mean subtracted, contrast normalized and whitened.


### One-hot vector

![](https://i.imgur.com/hO80OJr.png)


A vector containing one in a single entry and zero elsewhere. 


### Cross entropy

Commonly used to quantify the difference between two probability distributions. In the case of neural networks, one of the distributions is the output of the softmax, while the other is a one-hot vector corresponding to the correct class.

### Added noise

![](https://i.imgur.com/iywwP9W.png)


A perturbation added to the input of the network or one of the feature vectors it computes. 

---

## Datasets

### MNIST

![](https://i.imgur.com/2NT1JBR.png)


The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size 28x28 image.


### CIFAR-10

![](https://i.imgur.com/2AMdgSY.jpg)


The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

### ImageNet

![](https://i.imgur.com/GjZuvtj.png)


A large image database that has over ten million URLs of images that were hand-annotated using Amazon Mechanical Turk to indicate what objects are pictured.
[source](https://en.wikipedia.org/wiki/ImageNet)

---

## Contests

### ImageNet Large Scale Visual Recognition Challenge (ILSVRC)

![](https://i.imgur.com/3PNd24f.png)


A competition in which teams compete to obtain the highest accuracy on several computer vision tasks, such as image classification.
[source](https://en.wikipedia.org/wiki/ImageNet)



