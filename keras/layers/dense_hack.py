from tensorflow.keras.layers import InputSpec

def Dense_build(self, input_shape):
    assert len(input_shape) >= 2
    input_dim = input_shape[-1]

    self.kernel = self.add_weight(shape=(input_dim, self.units),
                                  initializer=self.kernel_initializer,
                                  name='%s_kernel' % self.name,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
    if self.use_bias:
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    name='%s_bias' % self.name,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
    else:
        self.bias = None
    self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
    self.built = True

from tensorflow.keras.layers import Dense
Dense.build = Dense_build
