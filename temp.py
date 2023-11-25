import numpy as np

# Activation function (tanh) and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

# Mean Squared Loss
def mean_squared_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Function to perform forward pass
def forward_pass(X, W1, b1, W2, b2):
    h = np.tanh(np.dot(X, W1) + b1)
    out = np.dot(h, W2) + b2
    return h, out

# Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initializing weights and biases with smaller values
lr = 0.1
np.random.seed(1)
W1 = np.random.randn(2, 4) * 0.1
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.1
b2 = np.zeros((1, 1))

# Training loop
for i in range(10001):
    # Forward Pass
    hidden, output = forward_pass(X, W1, b1, W2, b2)

    # Calculating and printing the loss
    loss = mean_squared_loss(y, output)
    if i % 1000 == 0:
        print(f"Epoch : {i} Loss : {loss}")

    # Backpropagation
    dL_dout = 2 * (output - y) / len(y)
    dL_dW2 = np.dot(hidden.T, dL_dout)
    dL_db2 = np.sum(dL_dout, axis=0, keepdims=True)
    dL_dh = np.dot(dL_dout, W2.T)
    dL_dz = dL_dh * tanh_derivative(hidden)
    dL_dW1 = np.dot(X.T, dL_dz)
    dL_db1 = np.sum(dL_dz, axis=0)

    # Updating parameters with Gradient descent
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2

# Printing expected output and model's output
print("Expected output:")
print(y)
print("Model's output:")
print(output)

