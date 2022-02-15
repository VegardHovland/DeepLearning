from cmath import sqrt
from turtle import dot
import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    
    my = np.mean(X) # mean pixel value
    sigma = np.std(X) # Standard deviation 
    X_norm = (X-my)/sigma
    batch_size, num_cols = X.shape
    bias = np.ones((batch_size, 1))
    X_norm = np.hstack((X_norm,bias))                   # Append collumn vector of ones to X

    return X_norm



def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    C = - np.sum(targets*np.log(outputs), axis=1)
    return np.mean(C)



class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        # self.ws = np.zeros((self.I, 1))  
        self.ws = []

        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            # w = np.zeros(w_shape) # only used for testing that fuctions are corectly implemented
            if use_improved_weight_init:
                w = np.random.normal(0, 1/np.sqrt(prev), w_shape)
            else: 
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]
        self.activations = np.array([None for i in range(len(self.ws))])

    # Sigmoid activation function, use between layers
    def sigmoid(self, z):
        if(self.use_improved_sigmoid):
            return 1.7159 * np.tanh(2/3 * z)
        else:
            return 1/(1 + np.exp(-z))

    def dSigmoid(self, z):
        if(self.use_improved_sigmoid):
            return 1.7159 * (4/3) / (np.cosh(z*4/3) + 1)
        else:
            return self.sigmoid(z)* (1 - self.sigmoid(z)) 

    # Softamx model for prediction in forward pass, objective function
    def softamx(self, z):
        s = np.divide(np.exp(z) , np.array([np.sum(np.exp(z),axis=1)]).T)
        return s

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...

        ### TASK 2 up to 4c
        # self.activations[0] = X                                                         # Input layer
        # self.activations[1] = self.sigmoid((X @ self.ws[0]))                            # Hidden 1
        # self.activations[2] = self.sigmoid(np.matmul(self.activations[1], self.ws[1]))  # output layer not sigmoid!!!
        # return self.softamx(self.activations[1] @ self.ws[-1])                          # Softamx for output layer

        ### Task 4c
        self.activations[0] = X
        for i in range(len(self.neurons_per_layer) - 1):                             # len of nerurons per layer is number of layers
            activation = self.sigmoid(np.matmul(self.activations[i], self.ws[i]))
            self.activations[i+1] = activation                                       # Python is not in place
        return self.softamx(activation @ self.ws[-1])
               

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """""
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer

        ### TASK 2 up to 4c
        ## Går bakover, først output layer
        # error0 = - (targets - outputs)
        #self.grads[1] =  np.dot(self.activations[1].T, (error0)) / targets.shape[0] # output layer
        ## Går bakover til hidden layer
        # error1 = self.dSigmoid(X @ self.ws[0]) * (error0 @ self.ws[1].T)     # Error for hidden layer
        # self.grads[0] = ((np.dot(X.T, (error1))) / X.shape[0])               # Hidden layer grads.   

        ### Task 4C!
        error = - (targets - outputs)
        self.grads[-1] =  np.dot(self.activations[-1].T, (error)) / targets.shape[0] # output layer
        for i in range((len(self.neurons_per_layer) - 1),  0, -1):
            error = self.dSigmoid(self.activations[i-1] @ self.ws[i-1]) * (error @ self.ws[i].T)     # Error for hidden layer
            self.grads[i-1] =  np.dot(self.activations[i-1].T, (error)) / targets.shape[0]

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    Y_var = np.zeros((Y.shape[0], num_classes), dtype=int)
    for n in range(Y.shape[0]):
        Y_var[n, Y[n]] = 1           # Create an one-hot encoded variable (1 for labeled variable, zero otherwise)
    return Y_var


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
