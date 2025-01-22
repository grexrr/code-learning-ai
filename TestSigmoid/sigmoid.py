import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

#initialize network parameters
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_1 = np.random.randn(input_size, hidden_size) * 0.1 # weights for the first layer(single hidden layer in this case)
    bias_1 = np.zeros((1, hidden_size))
    weights_2 = np.random.randn(hidden_size, output_size) * 0.1 # weights for the output layer
    bias_2 = np.zeros((1, output_size))
    return weights_1, bias_1, weights_2, bias_2

def forward_propagation(X, weights_1, bias_1, weights_2, bias_2):
    # hidden layer liniear transformation
    z1 = np.dot(X, weights_1) + bias_1
    # hidden layer non-linear transformation
    a1 = sigmoid(z1)
    # output layer linear transformation
    z2 = np.dot(a1, weights_2) + bias_2
    # output layer non-linear transformation
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def backward_propagation(X, y, z1, a1, z2, a2, weights_1, weights_2, bias_1, bias_2, learning_rate):
    # gradient of the output layer
    loss_grad = loss_derivative(y, a2) * sigmoid_derivative(z2)
    d_weights_2 = np.dot(a1.T, loss_grad)
    d_biased_2 = np.sum(loss_grad, axis=0, keepdims=True)

    # gradient of the hidden layer
    hidden_loss_grad = np.dot(loss_grad, weights_2.T) * sigmoid_derivative(z1)
    d_weights_1 = np.dot(X.T, hidden_loss_grad)
    d_bias_1 = np.sum(hidden_loss_grad, axis=0, keepdims=True)

    # iterate weights and biased
    weights_1 -= learning_rate * d_weights_1
    bias_1 -= learning_rate * d_bias_1
    weights_2 -= learning_rate * d_weights_2
    bias_2 -= learning_rate * d_biased_2
    
    return weights_1, bias_1, weights_2, bias_2

def train(X, y, input_size, hidden_size, output_size, epochs, learning_rate):
    weights_1, bias_1, weights_2, bias_2 = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        # forward propagation
        z1, a1, z2, a2 = forward_propagation(X, weights_1, bias_1, weights_2, bias_2)
        
        # loss calculation
        current_loss = loss(y, a2)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss}")
        
        # backpropagate
        weights_1, bias_1, weights_2, bias_2 = backward_propagation(
            X, y, z1, a1, z2, a2, weights_1, weights_2, bias_1, bias_2, learning_rate
        )
    
    return weights_1, bias_1, weights_2, bias_2

# test data
np.random.seed(42)
X = np.random.rand(100, 2)  # 100 data，each with 2 characteristics
y = np.array([[1 if x1 + x2 > 1 else 0 for x1, x2 in X]]).T  # 简单的二分类任务

# meta-parameter
input_size = 2  # characteristics number
hidden_size = 4  
output_size = 1  
epochs = 1000  
learning_rate = 0.1 

# train
trained_weights_1, trained_bias_1, trained_weights_2, trained_bias_2 = train(
    X, y, input_size, hidden_size, output_size, epochs, learning_rate
)

# test
_, _, _, predictions = forward_propagation(X, trained_weights_1, trained_bias_1, trained_weights_2, trained_bias_2)
predictions = (predictions > 0.5).astype(int)
accuracy = np.mean(predictions == y)
print(f"Final Accuracy: {accuracy * 100:.2f}%")