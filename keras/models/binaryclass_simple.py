from tensorflow import keras
from layers.simple import GarNet
from tensorflow_model_optimization.sparsity import keras as prune
def make_model(n_vert, n_feat, n_class=2):
    n_aggregators = 4
    n_filters = 8
    n_propagate = 8
    
    x = keras.layers.Input(shape=(n_vert, n_feat))
    n = keras.layers.Input(shape=(1,), dtype='int32')
    inputs = [x, n]
    pruning_params = {'pruning_schedule': prune.PolynomialDecay(initial_sparsity=0.30,final_sparsity=0.8, begin_step=1000,end_step=2734, frequency=100)}
    v = inputs
    v = [GarNet(4,8,8, input_format='xn', name='gar_1')(v),n]
    v =[GarNet(4,8,8, input_format='xn', name='gar_2')(v),n]
    v =GarNet(4,8,8, collapse='mean',input_format='xn', name='gar_3')(v)
    
    # pruning each layer#
    #v = [prune.prune_low_magnitude(GarNet(4,8,8, input_format='xn', name='gar_1'), **pruning_params)(v), n]
    #v = [prune.prune_low_magnitude(GarNet(4,8,8, input_format='xn', name='gar_2'), **pruning_params)(v), n]
    #v = prune.prune_low_magnitude(GarNet(n_aggregators, n_filters, n_propagate, collapse='mean', input_format='xn', name='gar_3'), **pruning_params)(v)
    if n_class == 2:
        v = keras.layers.Dense(1, activation='sigmoid')(v)
    else:
        v = keras.layers.Dense(1, activation='softmax')(v)
    outputs = v
    
    return keras.Model(inputs=inputs, outputs=outputs)

def make_loss(n_class=2):
    if n_class == 2:
        return 'binary_crossentropy'
    else:
        return 'categorical_crossentropy'


if __name__ == '__main__':
    import sys

    out_path = sys.argv[1]
    n_vert = int(sys.argv[2])
    n_feat = int(sys.argv[3])

    model = make_model(n_vert, n_feat, n_class=2)

    with open(out_path, 'w') as json_file:
        json_file.write(model.to_json())
