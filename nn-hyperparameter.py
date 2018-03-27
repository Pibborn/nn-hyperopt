from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sklearn.metrics import roc_auc_score


(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(int(x_train.shape[0]), 1, img_rows, img_cols)
    x_test = x_test.reshape(int(x_test.shape[0]), 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(int(x_train.shape[0]), img_rows, img_cols, 1)
    x_test = x_test.reshape(int(x_test.shape[0]), img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255



space = {'num_layers': hp.choice('num_layers', [2, 3]),

            'n_filters_1': hp.choice('n_filters_1', [8, 16, 32, 64]),
            'n_filters_2': hp.choice('n_filters_2', [8, 16, 32, 64]),
            'n_filters_3': hp.choice('n_filters_3', [16, 32, 64]),


            'drop_1': hp.uniform('drop_1', .25,.75),
            'drop_2': hp.uniform('drop_2',  .25,.75),

            'batch_size' : hp.choice('batch_size', [1, 2, 4, 8, 16, 32, 64]),

            'nb_epochs' :  2,
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'activation': 'relu'
        }


def run_model(param_dict):
  model = Sequential()
  model.add(Conv2D(param_dict['n_filters_1'], kernel_size=(3, 3),
                   activation='relu',
                   input_shape=input_shape))
  model.add(Conv2D(param_dict['n_filters_2'], (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(param_dict['drop_1']))
  model.add(Flatten())
  if param_dict['num_layers'] == 3:
    model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(Dense(param_dict['n_filters_3'], activation='relu'))
  model.add(Dropout(param_dict['drop_2']))
  model.add(Dense(10, activation='softmax'))

  model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=param_dict['optimizer'],
              metrics=['accuracy'])

  model.fit(x_val, y_val, nb_epoch=param_dict['nb_epochs'], batch_size=param_dict['batch_size'], verbose = 0)
  pred_auc =model.predict_proba(x_val, batch_size = 128, verbose = 0)
  acc = roc_auc_score(y_val, pred_auc)
  print('AUC:', acc)
  sys.stdout.flush()
  return {'loss': -[acc], 'status': STATUS_OK}

trials = Trials()
best = fmin(run_model, space, algo=tpe.suggest, max_evals=50, trials=trials)
