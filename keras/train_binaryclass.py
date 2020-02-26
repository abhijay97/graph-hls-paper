#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#import tensorflow as tf
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
##import keras
#from tensorflow.keras import backend as K
#K.set_session(sess)
import sys
from tensorflow import keras
from tensorflow.keras.callbacks import Callback,EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
import numpy as np
from models.binaryclass_simple import make_model, make_loss
from layers.simple import GarNet
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train the keras model.')
    parser.add_argument('--train', '-t', metavar='PATH', dest='train_path', nargs='+', help='Training data file.')
    parser.add_argument('--validate', '-v', metavar='PATH', dest='validation_path', nargs='+', help='Validation data file.')
    parser.add_argument('--out', '-o', metavar='PATH', dest='out_path', help='Write HDf5 output to path.')
    parser.add_argument('--ngpu', '-j', metavar='N', dest='ngpu', type=int, default=1, help='Use N GPUs.')
    parser.add_argument('--batch-size', '-b', metavar='N', dest='batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--num-epochs', '-e', metavar='N', dest='num_epochs', type=int, default=80, help='Number of epochs to train over.')
    parser.add_argument('--input-type', '-i', metavar='TYPE', dest='input_type', default='h5', help='Input data format (h5, root, root-sparse).')
    parser.add_argument('--generator', '-g', action='store_true', dest='use_generator', help='Use a generator for input.')
    parser.add_argument('--input-name', '-m', metavar='NAME', dest='input_name', default='events', help='Input dataset (TTree or HDF5 dataset) name.')

    args = parser.parse_args()
    del sys.argv[1:]

    n_class = 2
    n_vert_max = 256
    #features = list(range(6))
    #features = [0, 1, 2, 3]
    features = None
    #model2 = make_model(n_vert_max, n_feat=2, n_class=n_class)
   # model_file = 'results3/model_dense.json'
   # with open(model_file) as json_file:
    #    json_config = json_file.read()
    
    #model2 = keras.models.model_from_json(json_config, custom_objects = {'GarNet':GarNet})
    #model2.load_weights('results3/dense_weights.h5')
    #print(' actually loaded')
    #model2 = keras.models.load_model('results2/model_de.h5',custom_objects={'gar_1': GarNet, 'gar_2':GarNet})
    print('loaded')
    model = make_model(n_vert_max, n_feat=4, n_class=n_class)
    print(model.summary())
    #model.layers[5].set_weights(model2.layers[5].get_weights())
    model_json = model.to_json()
    with open("results3/model_dense.json", "w") as json_file:
        json_file.write(model_json)
        
    print('saving json')
    model_single = model
    if args.ngpu > 1:
        model = keras.utils.multi_gpu_model(model_single, args.ngpu)
    
    optimizer = keras.optimizers.Adam(lr=0.0005)
    
    model.compile(optimizer=optimizer, loss=make_loss(n_class), metrics=['acc'])

    if args.use_generator:
        if args.input_type == 'h5':
            from generators.h5 import make_generator
        elif args.input_type == 'root':
            from generators.uproot_fixed import make_generator
        elif args.input_type == 'root-sparse':
            from generators.uproot_jagged_keep import make_generator
        
        #train_gen,n_train = make_generator(args.train_path, args.batch_size, format='xn', features=features, n_vert_max=n_vert_max, y_dtype=np.float, y_features=None, dataset_name=args.input_name)
        train_gen, n_train_steps = make_generator(args.train_path, args.batch_size, features=features, n_vert_max=n_vert_max, dataset_name=args.input_name)
        #fit_kwargs['steps_per_epoch'] = n_train_steps
        print(train_gen)
        #n_train_steps = 2734 
        fit_kwargs = {'epochs': args.num_epochs, 'steps_per_epoch':n_train_steps}
        #print("\n\n\n here \n\n\n",n_train_steps)
        if args.validation_path:
            valid_gen, n_valid_steps = make_generator(args.validation_path, args.batch_size, features=features, n_vert_max=n_vert_max, dataset_name=args.input_name)
            fit_kwargs['validation_data'] = valid_gen
            fit_kwargs['validation_steps'] = n_valid_steps
        md_s = ModelCheckpoint('/afs/cern.ch/work/a/abgupta/graph_f/g5/graph-hls-paper/keras/results3/dense_weights.h5',save_best_only=True, save_weights_only=True, monitor='val_loss', model='min', verbose=1)
        callbacks = [EarlyStopping(monitor='val_loss',patience=7, verbose=1),ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1), md_s]
        fit_kwargs['callbacks'] = callbacks
        print(fit_kwargs)
        model.fit_generator(train_gen, **fit_kwargs)
        #print('qdense after train wt')
        #print(model.layers[3].get_weights()[0])
        model.reset_metrics()
        #model.save_weights('model_test.h5', save_format='h5')

    else:
        if args.input_type == 'h5':
            from generators.h5 import make_dataset
        elif args.input_type == 'root':
            from generators.uproot_fixed import make_dataset
        elif args.input_type == 'root-sparse':
            from generators.uproot_jagged_keep import make_dataset

        inputs, truth, shuffle = make_dataset(args.train_path[0], features=features, n_vert_max=n_vert_max, dataset_name=args.input_name)

        fit_kwargs = {'epochs': args.num_epochs, 'batch_size': args.batch_size, 'shuffle': shuffle}
        if args.validation_path:
            val_inputs, val_truth, _ = make_dataset(args.validation_path[0],format='xn', features=features, n_vert_max=n_vert_max, y_features=y_features, dataset_name=args.input_name)
            fit_kwargs['validation_data'] = (val_inputs, val_truth)
        md_s = ModelCheckpoint('/afs/cern.ch/work/a/abgupta/graph_f/g5/graph-hls-paper/keras/results3/dense_weights.h5',save_best_only=True, save_weights_only=True, monitor='val_loss', model='min', verbose=1)
        model.fit(inputs, truth, **fit_kwargs, callbacks=[md_s])
    
    #if args.out_path:
        #model.reset_metrics()
        #model.save(args.out_path)
        #model.save_weights(args.out_path, save_format='h5')
        #model.save_weights(filepath = args.out_path, save_format='h5')
        #keras.models.save_model(model,args.out_path, save_format='h5')
