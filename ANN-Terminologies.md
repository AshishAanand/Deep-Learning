# Artificial Neural Networks (ANN) Terminologies Cheat Sheet

## Table of Contents
- [Basic Concepts](#basic-concepts)
- [Network Architecture](#network-architecture)
- [Neuron Components](#neuron-components)
- [Learning Process](#learning-process)
- [Training Algorithms](#training-algorithms)
- [Activation Functions](#activation-functions)
- [Network Types](#network-types)
- [Performance Metrics](#performance-metrics)
- [Common Problems](#common-problems)
- [Implementation Tips](#implementation-tips)

---

## Basic Concepts

### **Artificial Neural Network (ANN)**
A computational model inspired by biological neural networks, consisting of interconnected processing nodes (neurons) that work together to solve problems.

### **Biological Inspiration**
ANNs mimic the structure and function of biological neurons in the brain, with artificial neurons receiving inputs, processing them, and producing outputs.

### **Machine Learning**
ANNs are a subset of machine learning that can learn patterns from data without being explicitly programmed for specific tasks.

### **Supervised Learning**
Training method where the network learns from input-output pairs (labeled data).

### **Unsupervised Learning**
Training method where the network finds patterns in data without labeled examples.

### **Reinforcement Learning**
Training method where the network learns through interaction with an environment using rewards and penalties.

---

## Network Architecture

### **Input Layer**
The first layer that receives external data/features and passes them to the network.
```
Input Layer: [x1, x2, x3, ..., xn]
```

### **Hidden Layer**
Intermediate layers between input and output that perform computations and feature extraction.
- **Single Hidden Layer**: Shallow network
- **Multiple Hidden Layers**: Deep network

### **Output Layer**
The final layer that produces the network's predictions or classifications.

### **Fully Connected Layer (Dense Layer)**
A layer where each neuron is connected to every neuron in the previous layer.

### **Layer Depth**
The number of layers in the network (including input, hidden, and output layers).

### **Network Width**
The number of neurons in each layer.

### **Topology**
The overall structure and arrangement of layers and connections in the network.

---

## Neuron Components

### **Artificial Neuron (Node)**
The basic processing unit that receives inputs, applies weights, adds bias, and produces an output.

### **Synaptic Weights (W)**
Numerical values that determine the strength of connections between neurons.
```python
# Weight matrix example
W = [[w11, w12, w13],
     [w21, w22, w23]]
```

### **Bias (b)**
An additional parameter that shifts the activation function, allowing for better fitting.
```python
# Neuron computation
output = activation_function(Σ(wi * xi) + b)
```

### **Weighted Sum**
The sum of all inputs multiplied by their corresponding weights plus the bias.
```
net_input = w1*x1 + w2*x2 + ... + wn*xn + b
```

### **Threshold**
A value that determines when a neuron should activate (mainly in step functions).

### **Connections**
Links between neurons that carry weighted signals from one layer to another.

---

## Learning Process

### **Training**
The process of adjusting weights and biases to minimize prediction errors.

### **Learning Rate (α)**
Hyperparameter that controls how much weights are adjusted during each update.
```python
new_weight = old_weight - learning_rate * gradient
```

### **Epoch**
One complete pass through the entire training dataset.

### **Iteration**
One update of the network parameters (can be per sample or per batch).

### **Convergence**
The point where the network's performance stops improving significantly.

### **Forward Propagation**
The process of passing input data through the network to generate predictions.
```
Input → Hidden Layer(s) → Output
```

### **Backward Propagation (Backpropagation)**
The process of calculating gradients and updating weights by propagating errors backward through the network.

### **Error Signal**
The difference between predicted and actual outputs, used to adjust weights.

---

## Training Algorithms

### **Gradient Descent**
Optimization algorithm that adjusts weights in the direction that reduces the error.
```python
# Basic gradient descent
weight = weight - learning_rate * (∂Error/∂weight)
```

### **Stochastic Gradient Descent (SGD)**
Gradient descent that updates weights after each training example.

### **Batch Gradient Descent**
Updates weights after processing the entire training dataset.

### **Mini-batch Gradient Descent**
Updates weights after processing small groups of training examples.

### **Delta Rule**
Learning rule for single-layer networks that adjusts weights based on prediction errors.

### **Generalized Delta Rule**
Extension of delta rule for multi-layer networks using backpropagation.

### **Momentum**
Technique that adds a fraction of the previous weight update to the current update.

### **Learning Schedule**
Strategy for adjusting the learning rate during training (e.g., decay over time).

---

## Activation Functions

### **Linear Activation**
`f(x) = x` - Produces output proportional to input (rarely used in hidden layers).

### **Step Function (Threshold)**
```
f(x) = 1 if x ≥ threshold
f(x) = 0 if x < threshold
```

### **Sigmoid Function**
`f(x) = 1/(1 + e^(-x))` - S-shaped curve, output between 0 and 1.
```python
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### **Hyperbolic Tangent (tanh)**
`f(x) = (e^x - e^(-x))/(e^x + e^(-x))` - Output between -1 and 1.

### **Rectified Linear Unit (ReLU)**
`f(x) = max(0, x)` - Zero for negative inputs, linear for positive inputs.

### **Softmax**
Used in output layer for multi-class classification, converts outputs to probabilities.
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
```

---

## Network Types

### **Feedforward Neural Network**
Information flows in one direction from input to output without cycles.

### **Perceptron**
Simplest ANN with single layer, can only solve linearly separable problems.

### **Multi-Layer Perceptron (MLP)**
Feedforward network with one or more hidden layers, can solve non-linear problems.

### **Recurrent Neural Network (RNN)**
Network with feedback connections, can process sequential data.

### **Radial Basis Function Network**
Uses radial basis functions as activation functions in hidden layer.

### **Self-Organizing Map (SOM)**
Unsupervised network that creates low-dimensional representations of input data.

### **Hopfield Network**
Recurrent network used for pattern recognition and associative memory.

---

## Performance Metrics

### **Mean Squared Error (MSE)**
Average of squared differences between predicted and actual values.
```python
MSE = (1/n) * Σ(y_actual - y_predicted)²
```

### **Root Mean Squared Error (RMSE)**
Square root of MSE, in same units as the target variable.

### **Mean Absolute Error (MAE)**
Average of absolute differences between predicted and actual values.

### **Accuracy**
Percentage of correct predictions (for classification problems).
```python
Accuracy = (Correct Predictions / Total Predictions) * 100
```

### **Precision**
Ratio of true positives to all positive predictions.

### **Recall (Sensitivity)**
Ratio of true positives to all actual positives.

### **F1-Score**
Harmonic mean of precision and recall.

---

## Common Problems

### **Overfitting**
When the network memorizes training data but fails to generalize to new data.
- **Symptoms**: High training accuracy, low validation accuracy
- **Solutions**: Regularization, dropout, early stopping

### **Underfitting**
When the network is too simple to learn the underlying patterns.
- **Symptoms**: Low training and validation accuracy
- **Solutions**: Increase network complexity, reduce regularization

### **Vanishing Gradient Problem**
Gradients become very small in deep networks, making learning slow or impossible.
- **Solutions**: Better activation functions (ReLU), normalization, skip connections

### **Exploding Gradient Problem**
Gradients become very large, causing unstable training.
- **Solutions**: Gradient clipping, better initialization, normalization

### **Local Minima**
The network gets stuck in suboptimal solutions during training.
- **Solutions**: Multiple random initializations, better optimizers, momentum

### **Slow Convergence**
The network takes too long to learn or doesn't improve significantly.
- **Solutions**: Adjust learning rate, use better optimizers, normalize inputs

---

## Implementation Tips

### **Data Preprocessing**
```python
# Normalize inputs to [0,1] or [-1,1]
X_normalized = (X - X.min()) / (X.max() - X.min())

# Standardize (zero mean, unit variance)
X_standardized = (X - X.mean()) / X.std()
```

### **Weight Initialization**
```python
# Random initialization
weights = np.random.randn(input_size, hidden_size) * 0.01

# Xavier initialization
weights = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
```

### **Learning Rate Selection**
- Start with 0.01 or 0.001
- Use learning rate schedules
- Monitor training loss

### **Network Size Guidelines**
- Start small and increase if needed
- Hidden layer size: between input and output size
- More layers for complex problems

---

## 🔧 Hyperparameters Reference

| Parameter | Common Values | Purpose |
|-----------|--------------|---------|
| Learning Rate | 0.001 - 0.1 | Controls weight update size |
| Hidden Layers | 1 - 5 | Network depth |
| Neurons per Layer | 10 - 1000 | Network width |
| Batch Size | 1 - 512 | Training efficiency |
| Epochs | 10 - 10000 | Training duration |
| Momentum | 0.9 - 0.99 | Accelerates learning |

## 📊 Activation Function Comparison

| Function | Range | Pros | Cons |
|----------|-------|------|------|
| Sigmoid | (0, 1) | Smooth, probabilistic | Vanishing gradient |
| Tanh | (-1, 1) | Zero-centered | Vanishing gradient |
| ReLU | [0, ∞) | No vanishing gradient | Dying neurons |
| Linear | (-∞, ∞) | Simple | Limited expressiveness |

## 🎯 Quick Start Checklist

- [ ] Preprocess and normalize your data
- [ ] Choose appropriate network architecture
- [ ] Select suitable activation functions
- [ ] Initialize weights properly
- [ ] Set reasonable learning rate
- [ ] Implement validation monitoring
- [ ] Plan for overfitting prevention

## 🧮 Essential Equations

**Neuron Output**:
```
y = f(Σ(wi * xi) + b)
```

**Error Calculation**:
```
E = (1/2) * Σ(target - output)²
```

**Weight Update**:
```
wi(new) = wi(old) - α * (∂E/∂wi)
```

**Sigmoid Derivative**:
```
f'(x) = f(x) * (1 - f(x))
```

---

*This cheat sheet covers fundamental ANN concepts and terminology. Perfect for students and practitioners getting started with artificial neural networks!*