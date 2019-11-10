import tensorflow as tf
from tensorflow import keras

class Slice_Layer(keras.layers.Layer):
    
    def __init__(self, output_dim, axis_start, axis_end, **kwargs):
        self.output_dim = output_dim
        self.axis_start = axis_start
        self.axis_end = axis_end
        super(Slice_Layer, self).__init__(**kwargs)
    
    def call(self, x):
        return x[:,self.axis_start:self.axis_end]
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = {'output_dim': self.output_dim, 'axis_start': self.axis_start, 'axis_end':self.axis_end}
        base_config = super(Slice_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))