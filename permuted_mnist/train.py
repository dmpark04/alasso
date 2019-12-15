
# coding: utf-8

import tensorflow as tf
slim = tf.contrib.slim
graph_replace = tf.contrib.graph_editor.graph_replace

import sys, os
sys.path.extend([os.path.expanduser('..')])
from alasso import utils

import numpy as np

# ## Parameters

# Data params
input_dim = 784
output_dim = 10

# Network params
n_hidden_units = 2000
activation_fn = tf.nn.relu

# Optimization params
batch_size = 256
epochs_per_task = 20 
learning_rate=1e-3

xi = 0.1
a_param = 1.8
a_prime = 1.0
epsilon = 1.0e-6
epsilon_prime = 0.0
omega_smoothing = 1.0

# Reset optimizer after each age
reset_optimizer = False


# ## Construct datasets
n_tasks = 30
n_repeat = 3

full_datasets, final_test_datasets = utils.construct_permute_mnist(num_tasks=n_tasks)
training_datasets = full_datasets
validation_datasets = final_test_datasets

# ## Construct network, loss, and updates
tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(n_hidden_units, activation=activation_fn, input_dim=input_dim))
model.add(Dense(n_hidden_units, activation=activation_fn))
model.add(Dense(output_dim, activation='softmax'))

from alasso import protocols
from alasso.optimizers import KOOptimizer
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import Callback
from alasso.keras_utils import LossHistory


protocol_name, protocol = protocols.ALASSO_PROTOCOL(a_param=a_param, a_prime=a_prime, epsilon=epsilon, epsilon_prime=epsilon_prime, omega_smoothing=omega_smoothing, xi=xi)
opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
opt_name = 'adam'
oopt = KOOptimizer(opt, model=model, **protocol)

model.compile(loss="categorical_crossentropy", optimizer=oopt, metrics=['accuracy'])
model.model._make_train_function()

history = LossHistory()
callbacks = [history]

file_prefix = "data_%s_opt%s_lr%.2e_bs%i_ep%i_tsks%i"%(protocol_name, opt_name, learning_rate, batch_size, epochs_per_task, n_tasks)
datafile_name = "%s.pkl.gz"%(file_prefix)


# ## Train!

diag_vals = dict()
all_evals = dict()

runtime_param_table = [
    (1.0, 1.0),
    ]

def run_fits(param_indices, training_data, valid_data, eval_on_train_set=False):
    for param_index in param_indices:
        current_params = runtime_param_table[param_index]

        fs = []
        evals = []
        print("setting params")
        print("param: %s"%str(current_params))

        for repeat in range(n_repeat):
            sess.run(tf.global_variables_initializer())

            oopt.set_c(current_params[0])
            oopt.set_c_prime(current_params[1])

            oopt.init_task_vars()

            print("    Repeat %i"%repeat)
            for age, tidx in enumerate(range(n_tasks)):
                print("        Age %i"%age)

                current_training_data_input = training_data[tidx][0]
                current_training_data_gt = training_data[tidx][1]
                shuffle = True

                oopt.set_nb_data(len(current_training_data_input))
                stuffs = model.fit(current_training_data_input, current_training_data_gt, batch_size, epochs_per_task, callbacks=callbacks,
                              verbose=0, shuffle=shuffle)
                oopt.update_task_metrics(current_training_data_input, current_training_data_gt, batch_size)
                oopt.update_task_vars()

                ftask = []
                for j in range(n_tasks):
                    if eval_on_train_set:
                        f_ = model.evaluate(training_data[j][0], training_data[j][1], batch_size, verbose=0)
                    else:
                        f_ = model.evaluate(valid_data[j][0], valid_data[j][1], batch_size, verbose=0)
                    ftask.append(np.mean(f_[1]) / float(n_repeat))
                if repeat == 0:
                    evals.append(ftask)
                else:
                    for j in range(n_tasks):
                        evals[tidx][j] = evals[tidx][j] + ftask[j]

                # Re-initialize optimizer variables
                if reset_optimizer:
                    oopt.reset_optimizer()

        evals = np.array(evals)
        all_evals[param_index] = evals
        
        # backup all_evals to disk
        utils.save_zipped_pickle(all_evals, datafile_name)

param_indices = list(range(len(runtime_param_table)))
print(param_indices)

# Run
run_fits(param_indices, training_datasets, validation_datasets)

# backup all_evals to disk
utils.save_zipped_pickle(all_evals, datafile_name)



