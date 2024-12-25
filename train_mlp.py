import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import all functions from our module
from mlp_model import (
    init_parameters,
    forward_propagation,
    cost_function,
    full_backward_propagation,
    update_parameters
)

# ------------------------------------
# 1) Load Data
# ------------------------------------
x_train = pd.read_csv('data/training_set.csv', header=None).values
y_train = pd.read_csv('data/training_labels_bin.csv', header=None).values

x_val = pd.read_csv('data/validation_set.csv', header=None).values
y_val = pd.read_csv('data/validation_labels_bin.csv', header=None).values

N = len(x_train)  # number of training samples
M = len(x_val)    # number of validation samples

num_feats = x_train.shape[1]
n_out = y_train.shape[1]

# ------------------------------------
# 2) Define NN Architecture & Hyperparams
# ------------------------------------
nn_arch = [
    {"input_dimension": num_feats, "output_dimension": 10, "activation_func": "sigmoid"},
    {"input_dimension": 10,        "output_dimension": 10, "activation_func": "sigmoid"},
    {"input_dimension": 10,        "output_dimension": 3,  "activation_func": "sigmoid"},
]

eta = 0.001       # initial learning rate
gamma = 0.00001   # learning rate decay multiplier
stepsize = 200
threshold = 0.0001
test_interval = 10
max_epoch = 500

# ------------------------------------
# 3) Initialize Parameters
# ------------------------------------
params_values = init_parameters(nn_arch, seed=2)

# For logging the SSE of each output node
train_total_1 = []
train_total_2 = []
train_total_3 = []

val_total_1 = []
val_total_2 = []
val_total_3 = []

# Start timing
start_time = time.time()

# ------------------------------------
# 4) Training Loop
# ------------------------------------
for epoch in range(max_epoch):
    order = np.random.permutation(N)
    sse_train = np.zeros((n_out, 1))
    
    for n in range(N):
        idx = order[n]
        X_in = x_train[idx].reshape((num_feats, 1))
        Y_in = y_train[idx].reshape((n_out, 1))
        
        # Forward pass
        Y_hat, cache = forward_propagation(nn_arch, params_values, X_in)
        
        # Calculate SSE for this sample
        sse_train += cost_function(Y_hat, Y_in)
        
        # Backward pass
        grads_values = full_backward_propagation(Y_hat, Y_in, cache, params_values, nn_arch)
        
        # Update parameters
        params_values = update_parameters(params_values, grads_values, nn_arch, eta)
    
    # Mean SSE for the epoch
    train_mse = sse_train / N
    
    train_total_1.append(train_mse[0])
    train_total_2.append(train_mse[1])
    train_total_3.append(train_mse[2])
    
    print(f"epoch: {epoch}; training cost: {train_mse.T}")
    
    # ------------------------------------
    # Validation step at intervals
    # ------------------------------------
    if epoch % test_interval == 0:
        print("--------------------- validation -----------------------")
        sse_val = np.zeros((n_out, 1))
        
        for v in range(M):
            X_val_in = x_val[v].reshape((num_feats, 1))
            Y_val_in = y_val[v].reshape((n_out, 1))
            
            Y_val_hat, _ = forward_propagation(nn_arch, params_values, X_val_in)
            sse_val += cost_function(Y_val_hat, Y_val_in)
        
        val_mse = sse_val / M
        
        val_total_1.append(val_mse[0])
        val_total_2.append(val_mse[1])
        val_total_3.append(val_mse[2])
        
        print(f"epoch: {epoch}; validation cost: {val_mse.T}")
        print("--------------------- validation end -----------------------")
        
        # Early stopping
        if (val_mse < threshold).all():
            print("Stopping early as validation error < threshold.")
            break
    
    # Decay the learning rate after 'stepsize' epochs
    if epoch % stepsize == 0 and epoch != 0:
        eta *= gamma
        print(f"Changed learning rate to: {eta}")

# ------------------------------------
# 5) Plotting
# ------------------------------------
train_total_1 = np.concatenate(train_total_1, axis=0)
train_total_2 = np.concatenate(train_total_2, axis=0)
train_total_3 = np.concatenate(train_total_3, axis=0)

val_total_1 = np.concatenate(val_total_1, axis=0)
val_total_2 = np.concatenate(val_total_2, axis=0)
val_total_3 = np.concatenate(val_total_3, axis=0)

# Plot training error (each output)
plt.figure()
plt.plot(range(len(train_total_1)), train_total_1, label='Output 1')
plt.plot(range(len(train_total_2)), train_total_2, label='Output 2')
plt.plot(range(len(train_total_3)), train_total_3, label='Output 3')
plt.xlabel('Epoch')
plt.ylabel('Squared Error')
plt.title('Training Error (Each Output)')
plt.legend()
plt.ylim(0, 0.0002)
plt.show()

# Plot validation error (each output)
plt.figure()
plt.plot(range(len(val_total_1)), val_total_1, label='Output 1')
plt.plot(range(len(val_total_2)), val_total_2, label='Output 2')
plt.plot(range(len(val_total_3)), val_total_3, label='Output 3')
plt.xlabel('Validation Step')
plt.ylabel('Squared Error')
plt.title('Validation Error (Each Output)')
plt.legend()
plt.ylim(0, 0.002)
plt.show()

# Plot training error (sum of outputs)
plt.figure()
plt.plot(range(len(train_total_1)), 
         train_total_1 + train_total_2 + train_total_3, 
         label='Sum of Outputs')
plt.xlabel('Epoch')
plt.ylabel('Squared Error')
plt.title('Training Error (Sum of Outputs)')
plt.ylim(0, 0.00035)
plt.legend()
plt.show()

# Plot validation error (sum of outputs)
plt.figure()
plt.plot(range(len(val_total_1)), 
         val_total_1 + val_total_2 + val_total_3, 
         label='Sum of Outputs')
plt.xlabel('Validation Step')
plt.ylabel('Squared Error')
plt.title('Validation Error (Sum of Outputs)')
plt.ylim(0, 0.004)
plt.legend()
plt.show()

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")
