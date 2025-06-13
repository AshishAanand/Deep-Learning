# Deep Learning Terminologies Cheat Sheet

## Table of Contents
- [Core Concepts](#core-concepts)
- [Neural Network Architecture](#neural-network-architecture)
- [Training Process](#training-process)
- [Optimization](#optimization)
- [Regularization](#regularization)
- [Loss Functions](#loss-functions)
- [Activation Functions](#activation-functions)
- [Model Evaluation](#model-evaluation)
- [Advanced Architectures](#advanced-architectures)
- [Hardware & Performance](#hardware--performance)

---

## Core Concepts

### **Artificial Neural Network (ANN)**
A computing system inspired by biological neural networks, consisting of interconnected nodes (neurons) that process information.

### **Deep Learning**
A subset of machine learning using neural networks with multiple hidden layers (typically 3+ layers) to learn complex patterns.

### **Perceptron**
The simplest form of neural network with a single layer, capable of learning linearly separable patterns.

### **Multi-Layer Perceptron (MLP)**
A feedforward neural network with multiple layers including input, hidden, and output layers.

### **Feature Learning**
The automatic discovery of representations needed for detection or classification from raw data.

---

## Neural Network Architecture

### **Layers**
- **Input Layer**: Receives raw data
- **Hidden Layer**: Intermediate processing layers
- **Output Layer**: Produces final predictions

### **Neurons/Nodes**
Individual processing units that receive inputs, apply weights, and produce outputs.

### **Weights (W)**
Parameters that determine the strength of connections between neurons.
```python
# Example weight matrix
W = [[0.2, 0.8], 
     [0.5, 0.1]]
```

### **Bias (b)**
Additional parameter added to weighted sum to shift the activation function.
```python
output = activation(W * input + b)
```

### **Feedforward**
The process of passing input data through the network from input to output layer.

### **Backpropagation**
Algorithm for training neural networks by propagating errors backward through the network.

---

## Training Process

### **Epoch**
One complete pass through the entire training dataset.

### **Batch**
A subset of training data used in one iteration of training.
- **Batch Size**: Number of samples in each batch
- **Mini-batch**: Small batch (typically 32-256 samples)
- **Stochastic**: Batch size of 1

### **Iteration**
One update of the model parameters (one forward and backward pass).

### **Forward Pass**
Computing predictions by passing data through the network.

### **Backward Pass**
Computing gradients and updating weights using backpropagation.

### **Gradient**
The derivative of the loss function with respect to parameters, indicating direction of steepest increase.

### **Learning Rate (α)**
Hyperparameter controlling how much to adjust weights during each update.
```python
new_weight = old_weight - learning_rate * gradient
```

---

## Optimization

### **Gradient Descent**
Basic optimization algorithm that updates parameters in the direction opposite to the gradient.

### **Stochastic Gradient Descent (SGD)**
Gradient descent using random samples instead of the full dataset.

### **Adam Optimizer**
Adaptive optimization algorithm that combines momentum and adaptive learning rates.

### **RMSprop**
Optimizer that adapts learning rate based on recent gradient magnitudes.

### **Momentum**
Technique that accelerates gradient descent by considering previous gradients.

### **Learning Rate Scheduling**
- **Step Decay**: Reduce learning rate at specific intervals
- **Exponential Decay**: Gradually reduce learning rate
- **Adaptive**: Adjust based on performance

---

## Regularization

### **Overfitting**
When a model performs well on training data but poorly on unseen data.

### **Underfitting**
When a model is too simple to capture underlying patterns in the data.

### **Dropout**
Randomly setting some neurons to zero during training to prevent overfitting.
```python
# Dropout with 50% probability
dropout_layer = tf.keras.layers.Dropout(0.5)
```

### **L1 Regularization (Lasso)**
Adds sum of absolute values of parameters to loss function.

### **L2 Regularization (Ridge)**
Adds sum of squared parameters to loss function.

### **Batch Normalization**
Normalizing inputs to each layer to stabilize and accelerate training.

### **Early Stopping**
Stopping training when validation performance stops improving.

---

## Loss Functions

### **Mean Squared Error (MSE)**
For regression tasks: `MSE = (1/n) * Σ(y_true - y_pred)²`

### **Cross-Entropy Loss**
For classification tasks: `CE = -Σ(y_true * log(y_pred))`

### **Binary Cross-Entropy**
For binary classification: `BCE = -(y*log(p) + (1-y)*log(1-p))`

### **Categorical Cross-Entropy**
For multi-class classification with one-hot encoded labels.

### **Sparse Categorical Cross-Entropy**
For multi-class classification with integer labels.

---

## Activation Functions

### **ReLU (Rectified Linear Unit)**
`f(x) = max(0, x)` - Most commonly used, solves vanishing gradient problem.

### **Sigmoid**
`f(x) = 1/(1 + e^-x)` - Output between 0 and 1, used in binary classification.

### **Tanh (Hyperbolic Tangent)**
`f(x) = (e^x - e^-x)/(e^x + e^-x)` - Output between -1 and 1.

### **Softmax**
`f(x_i) = e^x_i / Σe^x_j` - Used in multi-class classification output layer.

### **Leaky ReLU**
`f(x) = max(αx, x)` where α is small positive value - Addresses dying ReLU problem.

### **Swish**
`f(x) = x * sigmoid(x)` - Self-gated activation function.

---

## Model Evaluation

### **Validation Set**
Portion of data used to evaluate model during training.

### **Test Set**
Separate dataset used for final model evaluation.

### **Cross-Validation**
Technique to assess model generalization using multiple train/validation splits.

### **Confusion Matrix**
Table showing correct vs predicted classifications.

### **Precision**
`TP / (TP + FP)` - Accuracy of positive predictions.

### **Recall (Sensitivity)**
`TP / (TP + FN)` - Ability to find all positive instances.

### **F1-Score**
`2 * (Precision * Recall) / (Precision + Recall)` - Harmonic mean of precision and recall.

---

## Advanced Architectures

### **Convolutional Neural Network (CNN)**
Network using convolution operations, excellent for image processing.
- **Convolution Layer**: Applies filters to detect features
- **Pooling Layer**: Reduces spatial dimensions
- **Filter/Kernel**: Small matrix used for convolution

### **Recurrent Neural Network (RNN)**
Network with memory, processes sequential data.
- **LSTM**: Long Short-Term Memory - handles long sequences
- **GRU**: Gated Recurrent Unit - simplified LSTM
- **Vanishing Gradient**: Problem in deep RNNs

### **Transformer**
Architecture using self-attention mechanism for sequence processing.
- **Self-Attention**: Mechanism to focus on relevant parts of input
- **Multi-Head Attention**: Multiple attention mechanisms in parallel
- **Positional Encoding**: Adding position information to embeddings

### **Autoencoder**
Network that learns to compress and reconstruct data.
- **Encoder**: Compresses input to lower dimension
- **Decoder**: Reconstructs original input from compressed representation
- **Bottleneck**: Compressed representation layer

### **Generative Adversarial Network (GAN)**
Two networks competing: generator creates fake data, discriminator detects fake data.

---

## Hardware & Performance

### **GPU (Graphics Processing Unit)**
Specialized hardware for parallel processing, essential for deep learning.

### **TPU (Tensor Processing Unit)**
Google's custom chips optimized for tensor operations.

### **CUDA**
NVIDIA's parallel computing platform for GPU programming.

### **Mixed Precision Training**
Using both 16-bit and 32-bit floating-point representations to speed up training.

### **Data Parallel**
Training strategy distributing data across multiple devices.

### **Model Parallel**
Training strategy distributing model layers across multiple devices.

---

## 💡 Pro Tips

- **Start Simple**: Begin with basic architectures before moving to complex ones
- **Monitor Training**: Watch for overfitting using validation curves
- **Hyperparameter Tuning**: Learning rate is often the most important hyperparameter
- **Data Quality**: Good data is more valuable than complex models
- **Reproducibility**: Set random seeds for consistent results
- **Gradient Clipping**: Prevent exploding gradients in deep networks
- **Transfer Learning**: Use pre-trained models when possible

## 🔧 Common Hyperparameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| Learning Rate | 0.001 - 0.1 | Step size for parameter updates |
| Batch Size | 16 - 512 | Number of samples per batch |
| Epochs | 10 - 1000+ | Number of training iterations |
| Dropout Rate | 0.1 - 0.5 | Fraction of neurons to drop |
| Hidden Units | 64 - 2048 | Number of neurons per layer |

## 📚 Key Equations

**Gradient Descent Update**:
```
θ = θ - α * ∇J(θ)
```

**Backpropagation Chain Rule**:
```
∂L/∂w = (∂L/∂a) * (∂a/∂z) * (∂z/∂w)
```

**Softmax**:
```
σ(z_i) = e^z_i / Σ(e^z_j)
```

---

*This cheat sheet covers essential deep learning terminology. Keep it handy for quick reference during your ML journey!*