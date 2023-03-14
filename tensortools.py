#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auxiliary functions for Deep Learning with Tensorflow
Copyright (C) 2023  Henrique S. Xavier
Contact: hsxavier@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


def appropriate_initializer(activation, initializer=None):
    """
    Return the recommended NN weight initializer according to the 
    activation function.
    
    Input
    -----
    
    activation : str
        Neuron activation function. Possible options are:
        None, 'linear', 'tanh', 'sigmoid', 'softmax', 'relu', 'leaky_relu', 'prelu', 'selu'.
        
    initializer : str or keras Initializer obj
        Initializer to use. If set to None (default), this function will
        return a normal distribution with the recommended variance for `activation`.
        It can also be set to 'normal' or 'uniform', specifying the distribution to 
        be used, while the variance is set according to `activation`. Finally, if 
        anything else is passed, this function returns `initializer`.
        
    Return
    ------
    
    initializer : str
        The recommended initializer for the `activation` function, unless a specific
        `initializer` is set, in which case it is returned.
    """
    
    # Hard-coded:
    glorot_activations = [None, 'linear', 'tanh', 'sigmoid', 'softmax']
    he_activations     = ['relu', 'leaky_relu', 'prelu', 'elu']
    lecun_activations  = ['selu']
    all_activations    = glorot_activations + he_activations + lecun_activations
    
    # Block unknown input:
    assert activation in all_activations, 'Unknown activation function: ' + activation
    
    # Set distribution family (default to uniform):
    if initializer == None:
        dist = 'normal'
    if initializer in ('normal', 'uniform'):
        dist = initializer
        initializer = None
    
    # Set initialization according to Aurelien Geron p. 334:
    if initializer == None:
        if activation in glorot_activations:
            initializer = 'glorot_' + dist
        if activation in he_activations:
            initializer = 'he_' + dist
        if activation in lecun_activations:
            initializer = 'lecun_' + dist

    return initializer

class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

def init_dropout(mc=False, alpha=False, *args, **kwargs):
    if mc:
        if alpha:
            return MCAlphaDropout(*args, **kwargs)
        else:
            return MCDropout(*args, **kwargs)
    else:
        if alpha:
            return keras.layers.AlphaDropout(*args, **kwargs) 
        else:
            return keras.layers.Dropout(*args, **kwargs)

    
def build_dense_model(n_hidden=1, n_units=100, input_shape=(1,), activation='relu', 
                      n_outputs=1, output_activation=None, initializer=None, leak_alpha=0.3, batch_norm=False,
                      batch_norm_at_start=True, batch_norm_momentum=0.99, dropout_layers=0, dropout_rate=0.2, dropout_mc=False,
                      dropout_alpha=None, learning_rate=0.001, sgd_momentum=0.9, nesterov=True, adam=False, adam_beta_1=0.9, adam_beta_2=0.999,
                      loss='binary_crossentropy', metrics=None):
    """
    Build a fully-connected neural network with a single output.
    
    Input
    -----
    
    n_hidden : int (default 1)
        Number of hidden layers in the model.
        
    n_units : int (default 100)
        Number of neurons in each hidden layer.
        
    input_shape : tuple of ints (default (1,))
        Shape of the input, ignoring the number of examples (batch size). 
        For a dataset with X features, use `(X,)`. For examples that are 
        2D (e.g. X x Y images), use `(X, Y)`. Internally, examples with 
        dimension greater than 1 get flattened.
        PS: I believe this function only accepts... 
        
    activation : str (default 'relu')
        Activation function for the hidden units. It accepts strings for 
        advanced (not built-in) ones like leaky ReLU and PReLU.
    
    n_outputs : int (default 1)
        Number of outputs, i.e. number  of units (neurons) in the output layer.
    
    output_activation : str (default None)
        Activation function for the output layer. For regression, is should be 
        'linear' or None. For classification, it should be 'sigmoid' or 'softmax'
        ('softmax' makes sense only if `n_outputs` > 1).
        
    initializer : str, None or keras initializer (default None)
        Distribution used to sample hidden units initial weights. If unspecified 
        (`None`), set to a normal distribution with variance recommended by 
        Aurelien Geron p. 344 given the `activation` function. One can force 
        a normal distribution with 'normal' and a uniform distribution with 'uniform'.
        
    leak_alpha : float (default 0.3)
        The leak in case `activation` is 'leaky_relu'.
        
    batch_norm : bool or str (default False)
        How to apply Batch Normalization. It can be False (do not apply), 
        'before' (apply Batch Normalization right before using the activation 
        function), 'after' (apply Batch Normalization right after the activation 
        function), or True (defaults to 'before').
    
    batch_norm_at_start : bool (default True)
        If applying Batch Normalization, whether or not to apply to the input.
        
    batch_norm_momentum : float (default 0.99)
        Weight for the batch means and deviations when calculating the 
        exponential moving average.
    
    dropout_layers : int (default 0)
        Number of layers to apply dropout, starting from the last (top) 
        hidden layer. If `dropout_layers` is zero, do not apply dropout.
        Will use alpha dropout instead if necessary (check `dropout_alpha`
        parameter).
        
    dropout_rate : float (default 0.2)
        Rate at which each unit is dropped in case `dropout_layers` > 0.
    
    dropout_mc : bool (default False)
        Whether or not to use MC dropout (i.e. if during predicting time, 
        dropout should be applied as well).
        
    dropout_alpha : bool or None (default None)
        Whether or not to use AlphaDropout instead of Dropout. If None, 
        will default to False unless `activation` is 'selu'.
        
    learning_rate : float (default 0.001)
        The `learning_rate` parameter passed to the cost function optimizer.
        Its exact meaning depends on the optimizer used.
        
    sgd_momentum : float (default 0.9)
        The momentum in a Stochastic Gradient Descent (SGD) optimizer.
        0 means using the standard SGD (apart from the `nesterov` setting).
        1 means no friction.
    
    nesterov : bool (default True)
        Whether or not to use the Nesterov trick of computing the gradient 
        slightly ahead the current point in the parameter space when defining
        the next step.
        PS: if `adam` is True, this parameter selects between Nadam (`nesterov` 
        True) and Adam (`nesterov` False) optimizers.
        
    adam : bool (default False)
        Whether or not to use the Adam (or Nadam, in case `nesterov` is True) 
        optimizer.
        
    adam_beta_1 : float (default 0.9)
        Adam or Nadam optimizer's beta_1 parameter (the exponential moving 
        average weight of the gradient). 
    
    adam_beta_2 : float (default 0.999)
        Adam or Nadam optimizer's beta_2 parameter (the exponential moving 
        average weight of the gradient squared, used to supress the step in 
        the fastest declining direction). 
    
    loss : string or keras loss object (default 'binary_crossentropy')
        The loss function that will be minimized, with each step performed 
        with a different batch.
    
    metrics : list of keras metrics.
        Metrics to apply to the data after each epoch.
    
    Return
    ------
    
    model : keras model
        The model already compiled.
    """
    
    # Hard-coded:
    advanced_activations = ['leaky_relu', 'prelu']
    builtin_activations  = [None, 'linear', 'tanh', 'sigmoid', 'softmax', 'relu', 'selu', 'elu']
    batch_norm_options   = [True, False, 'before', 'after']
    
    # Derived variables:
    all_activations      = builtin_activations + advanced_activations

    # Block bad input:
    assert activation in all_activations, 'Unknown activation function: ' + activation
    assert batch_norm in batch_norm_options, 'Bad batch normalization specification: ' + batch_norm
    
    # Set defaults:
    if batch_norm == True:
        # Default batch normalization is right before the activation function, according to batch norm paper authors:
        batch_norm = 'before'
    if dropout_alpha == None: 
        # If not specified, use alpha dropout only if activation function is SELU:
        if activation == 'selu':
            dropout_alpha = True
        else:
            dropout_alpha = False
    
    # If initializer is not specified, set initializer based on activation function:
    initializer = appropriate_initializer(activation, initializer)
      
    # Create model:
    model = keras.models.Sequential()
    
    # Input layer:
    if len(input_shape) > 1:
        model.add(keras.layers.Flatten(input_shape=input_shape))
    else:
        model.add(keras.layers.Input(shape=input_shape))
        
    # Add batch normalization at start if requested it to be applied right before the activation function:
    if batch_norm == 'before' and batch_norm_at_start:
        model.add(keras.layers.BatchNormalization(momentum=batch_norm_momentum))
    
    if batch_norm == 'before':
        use_bias         = False
        dense_activation = None
    else:
        use_bias         = True
        
        if (activation in advanced_activations):
            dense_activation = None
        else:
            dense_activation = activation
        
    # Hidden layers:    
    for k in range(n_hidden):
        
        # Batch norm requested right after applying an activation function:
        if batch_norm == 'after':
            if (k != 0) or (k == 0 and batch_norm_at_start):
                model.add(keras.layers.BatchNormalization(momentum=batch_norm_momentum))
        
        # Add dense layer:
        if batch_norm == 'before' and k == 0 and batch_norm_at_start == False:
            # Add dropout to the `dropout_layers` layers:
            if n_hidden - k < dropout_layers:
                model.add(init_dropout(dropout_mc, dropout_alpha, rate=dropout_rate))
            model.add(keras.layers.Dense(n_units, activation=dense_activation, kernel_initializer=initializer, use_bias=True))
        else:
            # Add dropout to the `dropout_layers` layers:
            if n_hidden - k < dropout_layers:
                model.add(init_dropout(dropout_mc, dropout_alpha, rate=dropout_rate))
            model.add(keras.layers.Dense(n_units, activation=dense_activation, kernel_initializer=initializer, use_bias=use_bias))
        
        # Batch normalization right before activation function:
        if batch_norm == 'before':
            model.add(keras.layers.BatchNormalization(momentum=batch_norm_momentum))
        
        # In case of advanced activation functions:
        if activation in advanced_activations:
            if activation == 'leaky_relu':
                model.add(keras.layers.LeakyReLU(alpha=leak_alpha))
            elif activation == 'prelu':
                model.add(keras.layers.PReLU())
        
        # Built-in activation functions in case of batch normalizarion right before them:
        elif batch_norm == 'before':            
            model.add(keras.layers.Activation(activation))      
    
    # If requested, add batch normalization right after applying an activation function:
    if batch_norm == 'after':
        model.add(keras.layers.BatchNormalization(momentum=batch_norm_momentum))
    
    # Add dropout to the `dropout_layers` layers:
    if dropout_layers > 0:
        model.add(init_dropout(dropout_mc, dropout_alpha, rate=dropout_rate))
    # Output layer:
    model.add(keras.layers.Dense(n_outputs, activation=output_activation))

    # Compile the model
    if adam:
        if nesterov == True:
            opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=adam_beta_1, beta_2=adam_beta_2)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=adam_beta_1, beta_2=adam_beta_2)
    else:
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=sgd_momentum, nesterov=nesterov)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    
    return model


def dataset_to_numpy(dataset, process_X=None, y_scalar_to_array=False):
    """
    Parse a Tensorflow `dataset`, where each example is a 
    tuple (X, y) and X and y are tf.tensors, into X and y, 
    both as numpy arrays. If `y_scalar_to_array` is True,
    change the y shape so each example (that was a scalar) 
    is now 1D (the shape expected by keras models).

    If `process_X` is a function, apply it to each X instance
    imediately after transforming it to numpy array.
    """

    if type(process_X) == type(None):
        X = np.array([z.numpy() for z in dataset.map(lambda x, y: x)])
    else:
        X = np.array([process_X(z.numpy()) for z in dataset.map(lambda x, y: x)])
    y = np.array([z.numpy() for z in dataset.map(lambda x, y: y)])
    
    if y_scalar_to_array:
        y = y.reshape((len(y), 1))
    
    return X, y


class CollaborativeFilter(tf.Module):
    """
    Create a Collaborative Filter model for user ratings of items.

    The input data user-item matrix Y is factored into two matrices,
    one U representing the users and the other I, the items; both in
    a lower dimensional space:

    Y = U . It            (regression case)
    Y = sigmoid(U . It)   (binary classification case)

    Input
    -----

    input_shape : tuple of 2 ints
        The number of rows (users) and the number of columns
        (items) of the input user-item matrix Y. In the binary
        classification case, Y should contain only 1s, 0s or NaNs.

    n_dims : int
        The number of dimensions used to describe each user and item
        (the number of hidden or latent factors).

    learning_rate : float
        The learning rate of the steps in the Gradient Descent (GD).

    min_cost_reduction : float (default 0.0)
        A criterium to stop GD: the mininum fractional reduction in
        the cost function between two steps necessary to keep GD 
        running. When the fractional reduction reaches this value, 
        GD stops. 

    max_steps : int
        Maximum number of steps to perform in GD.

    binary : bool
        Whether the problem should be treated as classification (True) 
        or as regression (False). If True, Y is modeled by sigmoid(U . It)
        and the cost function is the cross entropy. If False, Y is 
        modeled by U . It and the cost funcion is the sum of squares.
    """
    
    def __init__(self, input_shape, n_dims, learning_rate=0.001, min_cost_reduction=0.0, max_steps=10000, binary=False):
        
        # Get input:
        self.input_shape = input_shape
        self.n_dims = n_dims
        self.learning_rate = learning_rate
        self.min_cost_reduction = min_cost_reduction
        self.max_steps = max_steps
        self.binary = binary

        # Security checks:
        assert type(binary) == bool, '`binary` should be boolean.'
        
        # Initialize internal variables:
        self.R     = tf.zeros(self.input_shape, dtype=tf.float32)
        self.Y     = tf.zeros(self.input_shape, dtype=tf.float32)
        self.delta = tf.Variable(tf.zeros(self.input_shape, dtype=tf.float32))
        if binary:
            self.probs = tf.Variable(tf.zeros(self.input_shape, dtype=tf.float32))
        self.n_users = self.input_shape[0]
        self.n_items = self.input_shape[1]
        self.items = tf.Variable(tf.random.normal((self.n_items, self.n_dims), dtype=tf.float32))
        self.users = tf.Variable(tf.random.normal((self.n_users, self.n_dims), dtype=tf.float32))
        self.last_cost = tf.Variable(np.inf)
        self.cost = tf.Variable(np.inf)
        self.grad_items = tf.Variable(tf.zeros_like(self.items))
        self.grad_users = tf.Variable(tf.zeros_like(self.users))
    
    @tf.function
    def __regression_training_loop__(self, user_item_matrix):
        """
        Use Gradient Descent and a Mean Squared Error cost function
        to approximate a `user_item_matrix` as a product of 
        two matrices.
        
        Output
        ------
               
        users : Variable Tensor. 
            The user matrix.

        items : Variable Tensor.
            The item matrix.

        cost  : Variable (scalar) Tensor.
            The final cost function value.
        """
        self.Y = tf.cast(user_item_matrix, dtype=tf.float32)
    
        # Create selection matrix R (recommender systems, check Andrew Ng's Machine Learning coursera):
        # This is one if there is an observation and 0 otherwise.
        self.R = tf.where(tf.math.is_nan(self.Y), 0.0, 1.0)
        # Fill nan in input to avoid crash:
        self.Y = tf.where(tf.math.is_nan(self.Y), 0.0, self.Y)

        # Training loop:
        for step in tf.range(self.max_steps):

            # Compute the cost function:
            self.delta = self.R * (self.users @ tf.transpose(self.items) - self.Y)
            self.cost  = tf.reduce_sum(tf.math.square(self.delta))

            # Compute the gradient:
            self.grad_users = self.delta @ self.items
            self.grad_items = tf.transpose(self.delta) @ self.users
            
            # Apply the gradient to the model's variables:
            self.users.assign_sub(self.learning_rate * self.grad_users)
            self.items.assign_sub(self.learning_rate * self.grad_items)
            
            # Stop if cost reduction is too small:
            if self.last_cost / self.cost - 1 < self.min_cost_reduction:
                break
            self.last_cost.assign(self.cost)

        return self.users, self.items, self.cost


    @tf.function
    def __classification_training_loop__(self, user_item_matrix):
        """
        Use Gradient Descent and a Cross Entropy cost function
        to approximate a `user_item_matrix` as the sigmoid of 
        a product of two matrices.
        
        Output
        ------
               
        users : Variable Tensor. 
            The user score matrix.

        items : Variable Tensor.
            The item score matrix.

        cost  : Variable (scalar) Tensor.
            The final cost function value.
        """
        self.Y = tf.cast(user_item_matrix, dtype=tf.float32)
    
        # Create selection matrix R (recommender systems, check Andrew Ng's Machine Learning coursera):
        # This is one if there is an observation and 0 otherwise.
        self.R = tf.where(tf.math.is_nan(self.Y), 0.0, 1.0)
        # Fill nan in input to avoid crash:
        self.Y = tf.where(tf.math.is_nan(self.Y), 0.0, self.Y)
                
        # Training loop:
        for step in tf.range(self.max_steps):

            # Compute the cost function:
            self.probs.assign(keras.activations.sigmoid(self.users @ tf.transpose(self.items)))
            self.cost.assign(-1.0 * tf.reduce_sum(self.R * ( self.Y * tf.math.log(tf.clip_by_value(self.probs, 1e-23, 1.0)) + (1.0 - self.Y) * tf.math.log(tf.clip_by_value(1.0 - self.probs, 1e-23, 1.0)) )))
            
            # Compute the gradient:
            self.delta.assign(self.R * (self.probs - self.Y))
            self.grad_users.assign(self.delta @ self.items)
            self.grad_items.assign(tf.transpose(self.delta) @ self.users)
            
            # Apply the gradient to the model's variables:
            self.users.assign_sub(self.learning_rate * self.grad_users)
            self.items.assign_sub(self.learning_rate * self.grad_items)
            
            # Stop if cost reduction is too small:
            if self.last_cost / self.cost - 1 < self.min_cost_reduction:
                break
            self.last_cost.assign(self.cost)

        return self.users, self.items, self.cost

    
    def fit(self, user_item_matrix):
        """
        Use Gradient Descent to approximate a 
        `user_item_matrix` as a product of two 
        matrices, `users` and `items` transposed, 
        or by the sigmoid of this product, if in 
        binary classification mode. These matrices 
        can be assessed as model attributes (after 
        fitting). The same happens with the final 
        `cost`. 

        The cost function is either Mean Squared Error or 
        Cross Entropy, depending on `binary`.

        Input
        -----
        
        user_item_matrix : Tensor, DataFrame or Array
            A 2D matrix with user ratings for items. The 
            users correspond to the rows and the items 
            to the columns. Unobserved ratings should be 
            NaNs.
        """

        if self.binary:
            self.users, self.items, self.cost = self.__classification_training_loop__(user_item_matrix)
        else:
            self.users, self.items, self.cost = self.__regression_training_loop__(user_item_matrix)
        
    
    def predict(self):
        """
        Return the model prediction for all users and items.
        That is, return `U . It` if `binary` is False and 
        `sigmoid(U . It)` if `binary` is True.
        """

        if self.binary:
            pred = keras.activations.sigmoid(self.users @ tf.transpose(self.items))
        else:
            pred = self.users @ tf.transpose(self.items)
        
        return pred

