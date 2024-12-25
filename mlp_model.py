import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_backward(dA, z):
    """
    Backward pass for sigmoid activation function.
    """
    sig = sigmoid(z)
    return dA * sig * (1.0 - sig)

def init_parameters(nn_arch, seed=99):
    """
    Initialize the weights and biases of the network.
    """
    np.random.seed(seed)
    parameters_values = {}
    
    for i, layer in enumerate(nn_arch, start=1):
        layer_input_size = layer['input_dimension']
        layer_output_size = layer['output_dimension']
        
        # Weight shape: (output_dimension, input_dimension)
        # Bias shape:   (output_dimension, 1)
        parameters_values[f'W{i}'] = np.random.randn(layer_output_size, layer_input_size)
        parameters_values[f'b{i}'] = np.random.randn(layer_output_size, 1)
    
    return parameters_values

def forward_propagation(nn_arch, parameters, input_value):
    """
    Perform one forward pass through the network.
    """
    cache = {}
    A_prev = input_value  # Activation of the previous layer
    
    # Cache the input as node_act0
    cache['node_act0'] = A_prev
    
    for i, layer in enumerate(nn_arch, start=1):
        W = parameters[f'W{i}']
        b = parameters[f'b{i}']
        
        # Linear step
        Z = np.dot(W, A_prev) + b
        # Non-linear activation
        A = sigmoid(Z)
        
        # Store intermediate values
        cache[f'node{i}'] = Z
        cache[f'node_act{i}'] = A
        
        A_prev = A  # Update for next layer
    
    Y_hat = A_prev
    return Y_hat, cache

def cost_function(y_pred, y_true):
    """
    Compute the element-wise squared error.
    """
    return np.square(y_true - y_pred)

def single_layer_backward(dA_curr, W_curr, b_curr, Z_curr, A_prev):
    """
    Compute backward propagation for a single layer.
    """
    dZ_curr = sigmoid_backward(dA_curr, Z_curr)   # shape: (output_dim, 1)
    dW_curr = np.dot(dZ_curr, A_prev.T)           # shape: (output_dim, input_dim)
    db_curr = dZ_curr                             # shape: (output_dim, 1)
    dA_prev = np.dot(W_curr.T, dZ_curr)           # shape: (input_dim, 1)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, cache, parameters, nn_arch):
    """
    Perform backward propagation for the entire network.
    """
    grads_values = {}
    Y = Y.reshape(Y_hat.shape)
    dA_prev = -2.0 * (Y - Y_hat)  # SSE derivative wrt Y_hat
    
    # Traverse layers in reverse
    for layer_idx_prev, layer in reversed(list(enumerate(nn_arch))):
        layer_idx_curr = layer_idx_prev + 1
        
        dA_curr = dA_prev
        A_prev = cache[f'node_act{layer_idx_prev}']
        Z_curr = cache[f'node{layer_idx_curr}']
        W_curr = parameters[f'W{layer_idx_curr}']
        b_curr = parameters[f'b{layer_idx_curr}']
        
        dA_prev, dW_curr, db_curr = single_layer_backward(
            dA_curr, W_curr, b_curr, Z_curr, A_prev
        )
        
        grads_values[f'dW{layer_idx_curr}'] = dW_curr
        grads_values[f'db{layer_idx_curr}'] = db_curr
    
    return grads_values

def update_parameters(parameters, grads_values, nn_arch, learning_rate):
    """
    Update network parameters (weights and biases).
    """
    for idx, _ in enumerate(nn_arch, start=1):
        parameters[f'W{idx}'] -= learning_rate * grads_values[f'dW{idx}']
        parameters[f'b{idx}'] -= learning_rate * grads_values[f'db{idx}']
    
    return parameters
