import numpy as np
from ffnn import FFNN

def get_numeric_gradient_differences(NN: FFNN, rows: np.ndarray, targets: np.ndarray, epsilon: float = 1e-7):
    weight_gradients, bias_gradients = [], []

    for layer in NN.layers:
        for weight_index, weight in enumerate(layer.w.flatten()):
            saved_weight = weight

            new_weights = layer.w.flatten()
            new_weights[weight_index] = saved_weight + epsilon
            layer.w = new_weights.reshape(layer.w.shape)
            w_plus_cost = NN.cost(targets, NN.forward(rows))

            new_weights = layer.w.flatten()
            new_weights[weight_index] = saved_weight - epsilon
            layer.w = new_weights.reshape(layer.w.shape)
            w_minus_cost = NN.cost(targets, NN.forward(rows))

            w_gradient_approx = (w_plus_cost - w_minus_cost) / (2 * epsilon)
            weight_gradients.append(w_gradient_approx)

            new_weights = layer.w.flatten()
            new_weights[weight_index] = saved_weight
            layer.w = new_weights.reshape(layer.w.shape)

        for bias_index, bias in enumerate(layer.b.flatten()):
            saved_bias = bias

            layer.b[0][bias_index] = saved_bias + epsilon
            b_plus_cost = NN.cost(targets, NN.forward(rows))

            layer.b[0][bias_index] = saved_bias - epsilon
            b_minus_cost = NN.cost(targets, NN.forward(rows))

            b_gradient_approx = (b_plus_cost - b_minus_cost) / (2 * epsilon)
            bias_gradients.append(b_gradient_approx)

            layer.b[0][bias_index] = saved_bias

    our_weight_gradients, our_bias_gradients = NN.backward(rows, targets) 
    our_weight_gradients = np.hstack(our_weight_gradients).ravel()
    our_bias_gradients = np.hstack(our_bias_gradients).ravel()

    weight_differences = np.array(weight_gradients) - our_weight_gradients
    bias_differences = np.array(bias_gradients) - our_bias_gradients
    return list(weight_differences), list(bias_differences), \
        {"numeric": weight_gradients, "backprop": list(our_weight_gradients)}, \
        {"numeric": bias_gradients, "backprop": list(our_bias_gradients)}