from __future__ import absolute_import, division, print_function, unicode_literals

import math
from tensorflow import keras
import tensorflow.keras.backend as K
from qkeras import *
import layers.dense_hack
from debug_flag import DEBUG
debug_summarize = None
from tensorflow_model_optimization.sparsity import keras as prune
#from tensorflow_model_optimization.python.core.sparsity.keras import prune
#from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
#ConstantSparsity = pruning_schedule.ConstantSparsity

class GarNet(keras.layers.Layer):
    def __init__(self, n_aggregators, n_filters, n_propagate, collapse=None, input_format='xn', discretize_distance=False, output_activation=None, mean_by_nvert=False, **kwargs):
        super(GarNet, self).__init__(**kwargs)

        self._setup_aux_params(collapse, input_format, discretize_distance, mean_by_nvert)
        self._setup_transforms(n_aggregators, n_filters, n_propagate, output_activation)

    def _setup_aux_params(self, collapse, input_format, discretize_distance, mean_by_nvert):
        if collapse is None:
            self.collapse = None
        elif collapse in ['mean', 'sum', 'max']:
            self.collapse = collapse
        else:
            raise NotImplementedError('Unsupported collapse operation')

        self.input_format = input_format
        self.discretize_distance = discretize_distance
        self.mean_by_nvert = mean_by_nvert

    def _setup_transforms(self, n_aggregators, n_filters, n_propagate, output_activation):
        pruning_params = {'pruning_schedule': prune.PolynomialDecay(initial_sparsity=0.50,final_sparsity=0.9, begin_step=2000,end_step=2734, frequency=100)}
        #pruning_params = {'pruning_schedule': ConstantSparsity(0.75, begin_step=2000, frequency=100)}
        #self.input_feature_transform = prune.prune_low_magnitude(QDense(n_propagate, kernel_quantizer=binary(1),bias_quantizer=binary(1), name='FLR'), **pruning_params)
        #self.aggregator_distance = keras.layers.Dense(n_aggregators, name='S')
        #self.output_feature_transform = prune.prune_low_magnitude(QDense(n_filters, kernel_quantizer=binary(1),bias_quantizer=binary(1), name='Fout'), **pruning_params)
        self.input_feature_transform = prune.prune_low_magnitude(keras.layers.Dense(n_propagate, name='FLR'), **pruning_params)
        #self.input_feature_transform = keras.layers.Dense(n_propagate, name='FLR')
        self.aggregator_distance = keras.layers.Dense(n_aggregators, name='S')
        self.output_feature_transform = keras.layers.Dense(n_filters, activation=output_activation, name='Fout')

        self._sublayers = [self.input_feature_transform, self.aggregator_distance, self.output_feature_transform]

    def build(self, input_shape):
        super(GarNet, self).build(input_shape)

        if self.input_format == 'x':
            data_shape = input_shape
        elif self.input_format == 'xn':
            data_shape, _ = input_shape
        elif self.input_format == 'xen':
            data_shape, _, _ = input_shape
            data_shape = data_shape[:2] + (data_shape[2] + 1,)

        self._build_transforms(data_shape)

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)

    def _build_transforms(self, data_shape):
        self.input_feature_transform.build(data_shape)
        self.aggregator_distance.build(data_shape)
       # print("agg dist and ip ft tran",self.aggregator_distance.units, self.input_feature_transform.units)
        self.output_feature_transform.build(data_shape[:2] + (4 * 8,))

    def call(self, x):
        data, num_vertex, vertex_mask = self._unpack_input(x)

        output = self._garnet(data, num_vertex, vertex_mask,
                              self.input_feature_transform,
                              self.aggregator_distance,
                              self.output_feature_transform)

        output = self._collapse_output(output)

        return output

    def _unpack_input(self, x):
        if self.input_format == 'x':
            data = x

            vertex_mask = K.cast(K.not_equal(data[..., 3:4], 0.), 'float32')
            num_vertex = K.sum(vertex_mask)

        elif self.input_format in ['xn', 'xen']:
            if self.input_format == 'xn':
                data, num_vertex = x
            else:
                data_x, data_e, num_vertex = x
                data = K.concatenate((data_x, K.reshape(data_e, (-1, data_e.shape[1], 1))), axis=-1)
    
            if DEBUG:
                data = K.print_tensor(data, message='data is ', summarize=debug_summarize)
                num_vertex = K.print_tensor(num_vertex, message='num_vertex is ', summarize=debug_summarize)
    
            data_shape = K.shape(data)
            B = data_shape[0]
            V = data_shape[1]
            vertex_indices = K.tile(K.expand_dims(K.arange(0, V), axis=0), (B, 1)) # (B, [0..V-1])
            vertex_mask = K.expand_dims(K.cast(K.less(vertex_indices, K.cast(num_vertex, 'int32')), 'float32'), axis=-1) # (B, V, 1)
            num_vertex = K.cast(num_vertex, 'float32')

        if DEBUG:
            vertex_mask = K.print_tensor(vertex_mask, message='vertex_mask is ', summarize=debug_summarize)

        return data, num_vertex, vertex_mask

    def _garnet(self, data, num_vertex, vertex_mask, in_transform, d_compute, out_transform):
        features = in_transform(data) # (B, V, F)
        distance = d_compute(data) # (B, V, S)

        if DEBUG:
            features = K.print_tensor(features, message='features is ', summarize=debug_summarize)
            distance = K.print_tensor(distance, message='distance is ', summarize=debug_summarize)

        if self.discretize_distance:
            distance = K.round(distance)
            if DEBUG:
                distance = K.print_tensor(distance, message='rounded distance is ', summarize=debug_summarize)

        edge_weights = vertex_mask * K.exp(K.square(distance) * (-math.log(2.))) # (B, V, S)
        
        if DEBUG:
            edge_weights = K.print_tensor(edge_weights, message='edge_weights is ', summarize=debug_summarize)

        if self.mean_by_nvert:
            def graph_mean(out, axis):
                s = K.sum(out, axis=axis)
                # reshape just to enable broadcasting
                s = K.reshape(s, (-1, d_compute.units * in_transform.units)) / num_vertex
                s = K.reshape(s, (-1, d_compute.units, in_transform.units))
                return s
        else:
            graph_mean = K.mean

        # vertices -> aggregators
        edge_weights_trans = K.permute_dimensions(edge_weights, (0, 2, 1)) # (B, S, V)
        aggregated = self._apply_edge_weights(features, edge_weights_trans, aggregation=graph_mean) # (B, S, F)

        if DEBUG:
            aggregated = K.print_tensor(aggregated, message='aggregated is ', summarize=debug_summarize)

        # aggregators -> vertices
        updated_features = self._apply_edge_weights(aggregated, edge_weights) # (B, V, S*F)

        if DEBUG:
            updated_features = K.print_tensor(updated_features, message='updated_features is ', summarize=debug_summarize)

        return vertex_mask * out_transform(updated_features)

    def _collapse_output(self, output):
        if self.collapse == 'mean':
            if self.mean_by_nvert:
                output = K.sum(output, axis=1) / num_vertex
            else:
                output = K.mean(output, axis=1)
        elif self.collapse == 'sum': 
           output = K.sum(output, axis=1)
        elif self.collapse == 'max':
            output = K.max(output, axis=1)

        if DEBUG:
            output = K.print_tensor(output, message='output is ', summarize=-1)

        return output

    def compute_output_shape(self, input_shape):
        return self._get_output_shape(input_shape, self.output_feature_transform)

    def _get_output_shape(self, input_shape, out_transform):
        if self.input_format == 'x':
            data_shape = input_shape
        elif self.input_format == 'xn':
            data_shape, _ = input_shape
        elif self.input_format == 'xen':
            data_shape, _, _ = input_shape

        if self.collapse is None:
            return data_shape[:2] + (out_transform.units,)
        else:
            return (data_shape[0], out_transform.units)

    def get_config(self):
        config = super(GarNet, self).get_config()

        config.update({
            'collapse': self.collapse,
            'input_format': self.input_format,
            'discretize_distance': self.discretize_distance,
            'mean_by_nvert': self.mean_by_nvert
        })

        self._add_transform_config(config)

        return config

    def _add_transform_config(self, config):
        config.update({
            'n_aggregators': self.aggregator_distance.units,
            'n_filters': self.output_feature_transform.units,
            'n_propagate': self.input_feature_transform.units
        })

    @staticmethod
    def _apply_edge_weights(features, edge_weights, aggregation=None):
        features = K.expand_dims(features, axis=1) # (B, 1, v, f)
        edge_weights = K.expand_dims(edge_weights, axis=3) # (B, u, v, 1)

        if DEBUG:
            features = K.print_tensor(features, message='applying on features ', summarize=debug_summarize)
            edge_weights = K.print_tensor(edge_weights, message='applying weights ', summarize=debug_summarize)

        out = edge_weights * features # (B, u, v, f)

        if DEBUG:
            out = K.print_tensor(out, message='before aggregation ', summarize=debug_summarize)

        if aggregation:
            out = aggregation(out, axis=2) # (B, u, f)
        else:
            out = K.reshape(out, (-1, edge_weights.shape[1], features.shape[-1] * features.shape[-2]))  # for keras = .shape[].value
            
        return out
