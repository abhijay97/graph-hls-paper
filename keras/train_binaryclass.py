#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tensorflow as tf
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#import keras
#from keras import backend as K
#K.set_session(sess)
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN


import sys
#import keras

from models.binaryclass_simple import make_model, make_loss

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

    model = make_model(n_vert_max, n_feat=4, n_class=n_class)
    model_single = model
    if args.ngpu > 1:
        model = keras.utils.multi_gpu_model(model_single, args.ngpu)
    
    optimizer = keras.optimizers.Adam(lr=0.005)
    
    model.compile(optimizer=optimizer, loss=make_loss(n_class),metrics=['acc'])
    print(model.summary())
    model_json = model.to_json()
    with open("model_binary_wtb.json", "w") as json_file:
        json_file.write(model_json)

    if args.use_generator:
        if args.input_type == 'h5':
            from generators.h5 import make_generator
        elif args.input_type == 'root':
            from generators.uproot_fixed import make_generator
        elif args.input_type == 'root-sparse':
            from generators.uproot_jagged_keep import make_generator

        train_gen, n_train_steps = make_generator(args.train_path, args.batch_size, features=features, n_vert_max=n_vert_max, dataset_name=args.input_name)
       # print(' this is train gen', train_gen)
        fit_kwargs = {'steps_per_epoch': n_train_steps, 'epochs': args.num_epochs}

        if args.validation_path:
            valid_gen, n_valid_steps = make_generator(args.validation_path, args.batch_size, features=features, n_vert_max=n_vert_max, dataset_name=args.input_name)
            fit_kwargs['validation_data'] = valid_gen
            fit_kwargs['validation_steps'] = n_valid_steps

        model.fit_generator(train_gen, **fit_kwargs, callbacks=[EarlyStopping(monitor='val_loss',patience=5),ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)])

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
            val_inputs, val_truth, _ = make_dataset(args.validation_path[0], format=input_format, features=features, n_vert_max=n_vert_max, y_features=y_features, dataset_name=args.input_name)
            fit_kwargs['validation_data'] = (val_inputs, val_truth)

        model.fit(inputs, truth, **fit_kwargs)
    
    if args.out_path:
        model_single.save(args.out_path)
