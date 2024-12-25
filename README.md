MLP with Backpropagation (NumPy Only)

```mermaid
flowchart LR
    subgraph Input Layer
        x1((x₁))
        x2((x₂))
        x3((x₃))
        xN((xₙ))
    end

    subgraph Hidden Layer 1
        h1((H1₁))
        h2((H1₂))
        h3((H1₃))
    end
    
    subgraph Hidden Layer 2
        h4((H2₁))
        h5((H2₂))
        h6((H2₃))
    end
    
    subgraph Output Layer
        y1((y₁))
        y2((y₂))
        y3((y₃))
    end

    %% Connections from Input to Hidden Layer 1
    x1 --> h1
    x1 --> h2
    x1 --> h3

    x2 --> h1
    x2 --> h2
    x2 --> h3

    x3 --> h1
    x3 --> h2
    x3 --> h3

    xN --> h1
    xN --> h2
    xN --> h3

    %% Connections from Hidden Layer 1 to Hidden Layer 2
    h1 --> h4
    h1 --> h5
    h1 --> h6

    h2 --> h4
    h2 --> h5
    h2 --> h6

    h3 --> h4
    h3 --> h5
    h3 --> h6

    %% Connections from Hidden Layer 2 to Output Layer
    h4 --> y1
    h4 --> y2
    h4 --> y3

    h5 --> y1
    h5 --> y2
    h5 --> y3

    h6 --> y1
    h6 --> y2
    h6 --> y3 ```

This project is a minimal implementation of a Multi-Layer Perceptron (MLP) using only Python and NumPy. It demonstrates how to build a neural network from scratch, including initialization, forward propagation, backpropagation, and weight updates.
Overview

    Goal: Implement an MLP (with two hidden layers) to classify or regress data using only NumPy—no deep learning frameworks (e.g., PyTorch, TensorFlow).
    Key Features:
        Two hidden layers: The architecture is input→hidden1→hidden2→outputinput→hidden1→hidden2→output.
        Sigmoid activation in all layers.
        Output layer has 3 neurons (3-dimensional output).
        Batch size = 1 for simplicity.
        Loss function: Sum of Squared Errors (SSE) across outputs.
        Learning rate decay: Adjusts every specified number of epochs.
        Early stopping: Monitors validation error against a threshold.

Project Structure

.
├── data/
│   ├── training_labels_bin.csv
│   └── validation_labels_bin.csv
├── mlp_model.py
├── train_mlp.py
└── README.md

    data/: Contains the CSV files for:
        Training set (training_set.csv)
        Training labels (training_labels_bin.csv)
        Validation set (validation_set.csv)
        Validation labels (validation_labels_bin.csv)

    mlp_model.py:
        Contains the function definitions for all core MLP operations:
            Initialization of weights and biases (init_parameters).
            Forward propagation (forward_propagation).
            Sigmoid activation (sigmoid, sigmoid_backward).
            Loss function (SSE, cost_function).
            Backward propagation (full_backward_propagation, single_layer_backward).
            Parameter updates (update_parameters).

    train_mlp.py:
        Loads the CSV data using pandas.
        Defines hyperparameters (learning rate, decay rate, threshold, etc.).
        Defines the network architecture (two hidden layers, each with 10 hidden units, followed by 3 output units).
        Trains the MLP with a loop over epochs.
            Forward Pass → Calculate Error → Backward Pass → Update Weights.
            Shuffles the data every epoch.
        Computes and prints the SSE metrics for training and validation.
        Plots the training/validation errors for each output node and for the sum of all outputs.

Requirements

    Python 3.7+
    NumPy
    pandas
    matplotlib

Install requirements with:

pip install numpy pandas matplotlib

Usage

    Prepare Data:
        Ensure training_set.csv, training_labels_bin.csv, validation_set.csv, and validation_labels_bin.csv are located in the data/ folder.
        Each .csv file should match the format used by this script (rows = samples, columns = features/labels).

    Run Training:

    python train_mlp.py

        The script will:
            Load the training and validation sets.
            Initialize the MLP parameters randomly.
            Train the MLP via backpropagation.
            Print epoch-wise SSE metrics.
            Plot error curves.

    Check the Output:
        The script will display plots of the training and validation errors for each output dimension and for the sum of outputs.
        You can view SSE values for each epoch in the console.

How It Works

    Initialization (init_parameters):
        We assign random values to each layer’s weight matrix and bias vector.

    Forward Pass (forward_propagation):
        For each layer, compute Z=W×Aprev+bZ=W×Aprev​+b and apply the Sigmoid activation: A=σ(Z)A=σ(Z).
        The final layer’s activation is the network’s output.

    Loss Calculation (cost_function):
        Computes SSE=∑(y−y^)2SSE=∑(y−y^​)2.
        During training, we accumulate the SSE over samples and then average it per epoch.

    Backward Pass (full_backward_propagation):
        The derivative of the SSE loss w.r.t. the network output is −2(y−y^)−2(y−y^​).
        We backpropagate through each layer using chain rule and the Sigmoid derivative.

    Parameter Update (update_parameters):
        We use simple gradient descent: θ←θ−η⋅∇θθ←θ−η⋅∇θ​.
        Optionally, the learning rate decays after a set number of epochs.

    Validation & Early Stopping:
        Every test_interval epochs, we compute validation SSE.
        If the validation SSE for all output nodes is below a threshold, we stop training.

Customization

    Hidden Layers: Change the nn_arch in train_mlp.py to add/remove layers or adjust hidden units.
    Activation: If you need a different activation (e.g., ReLU), replace sigmoid and sigmoid_backward.
    Loss Function: You can replace SSE with another metric (e.g., cross-entropy) if you want.
    Learning Rate Schedule: Modify how often (and by how much) the learning rate decays.

Contributing

    Feel free to fork this project, create issues, and open pull requests.
    For major changes, please discuss in an issue first.

License

Include the license that applies to your project (e.g., MIT, Apache, or any institutional policy).

Happy Training!
