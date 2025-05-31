# Elegans AI - Model 2 - Copyright Prof. Francesco Bardozzo 
# Source Code Distributed under Apache Licence 2.0
# Contacts fbardozzo@unisa.it
#@article{BARDOZZO2024127598,
#title = {Elegans-AI: How the connectome of a living organism could model artificial neural networks},
#journal = {Neurocomputing},
#volume = {584},
#pages = {127598},
#year = {2024},
#issn = {0925-2312},
#doi = {https://doi.org/10.1016/j.neucom.2024.127598},
#url = {https://www.sciencedirect.com/science/article/pii/S0925231224003692},
#author = {Francesco Bardozzo and Andrea Terlizzi and Claudio Simoncini and Pietro Lió and Roberto Tagliaferri}
#}


import igraph
import tensorflow.keras.callbacks
import tensorflow as tf
import tensorflow.keras.utils
import gc

from tensorflow.keras.layers import Layer, Dense, Lambda, Dot, Activation, Concatenate,Dense, Input,MultiHeadAttention
from tensorflow.keras.callbacks import TensorBoard
from numpy.random import seed
from collections import Counter
from datetime import datetime


from tensorflow.keras import regularizers

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import os
from os import path
import cv2
import csv
import sys
import re

from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange


from igraph import *
import numpy as np
from Connectome_Reader import Connectome_Reader

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import pydot
import graphviz


#####################################################
# Several stats
# Define save directory
parser = argparse.ArgumentParser(description='Elegans AI - Copyright Francesco Bardozzo - Source Code Distributed under Apache Licence 2.0')
parser.add_argument('--ratio',                     type=int,     help="Dense Layers times Ratio",   default = 1)
parser.add_argument('--deep_neurons',              type=int,     help='unit neurons',               default = 512)
parser.add_argument('--batch_size',                type=int,     help='batch size',                 default = 128)
parser.add_argument('--num_epochs',                type=int,     help='number of epochs',           default = 1000)
parser.add_argument('--seed',                      type=int,     help='seed',                       default = 6386)
parser.add_argument('--learning_rate',             type=float,   help='Learning Rate %%%',          default = 0.001) #gli ho messo uno zero in più
parser.add_argument('--min_learning_rate', type=float,   help='(unused placeholder)', default=0.0)
parser.add_argument('--gpus',                      type=int,     help='The number of GPUs to use',  default = 1 )
parser.add_argument('--gpuids',                    type=str,     help='IDs of GPUs to use 0,1,2,3', default = '0,1,2,3')
parser.add_argument('--model_code',                type=str,     help='model code number',          default = "original")
parser.add_argument('--path_to_connectome',        type=str,     help='path_to_connectome',         default = './celegans/celegans.graphml')
#parser.add_argument('--path_to_connectome',        type=str,     help='path_to_connectome',         default = './celegans/graph_clique_test.graphml')
parser.add_argument('--subd',                      type=str,     help='subd',                       default = "org")
parser.add_argument('--save_to',                   type=str,     help='save_to',                    default = './mnist_autoenc/')

args = parser.parse_args()

# Inform about multi-gpu training
if args.gpus == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
    print('-------  Will use GPU ' + args.gpuids)
    #gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

else:
    print('-------- Will use ' + str(args.gpus) + ' GPUs.')

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('--- Number of devices: {}'.format(strategy.num_replicas_in_sync))



ratio         = args.ratio
deep_neurons  = args.deep_neurons
batch_size    = args.batch_size
num_epochs    = args.num_epochs
seed          = args.seed
learning_rate = args.learning_rate
min_learning_rate = args.min_learning_rate
gpus          = args.gpus
gpuids        = args.gpuids
model_code    = args.model_code

path_to_connectome=args.path_to_connectome
subd          = args.subd
save_to       = args.save_to

#Set the seed for reproducibility
seed_value = 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
#tf.keras.utils.set_random_seed(seed_value)

print("Tensorflow version:")
print(tf.__version__)


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

else:
   print("Please install GPU version of TF")
   exit()

class check_gradient(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        #print(self.model.trainable_variables)
        self.model.trainable_variables

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def exists(val):
    return val is not None

class F1Metric(tf.keras.metrics.Metric):
    """
    Computes (micro) F1-score for binary data, given a fixed threshold.

    - y_true is expected to be floats in [0,1], effectively 0/1 after preprocessing.
    - y_pred is a float in [0,1]. We compare (y_pred >= threshold) to decide positive class.
    """

    def __init__(self, threshold=0.5, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        # Track true positives, false positives, false negatives
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: shape [batch_size, ...], floats 0/1
        # y_pred: shape [batch_size, ...], floats [0,1]
        # Binarize predictions & truths
        y_pred_bin = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true_bin = tf.cast(y_true >= 0.5, tf.float32)

        # Flatten everything to compute counts over all pixels/elements
        y_pred_bin = tf.reshape(y_pred_bin, [-1])
        y_true_bin = tf.reshape(y_true_bin, [-1])

        # True positives = predicted=1 & true=1
        tp = tf.reduce_sum(y_true_bin * y_pred_bin)
        # False positives = predicted=1 & true=0
        fp = tf.reduce_sum((1.0 - y_true_bin) * y_pred_bin)
        # False negatives = predicted=0 & true=1
        fn = tf.reduce_sum(y_true_bin * (1.0 - y_pred_bin))

        # Accumulate
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        # precision = TP / (TP + FP), recall = TP / (TP + FN)
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        # F1 = 2 * (precision * recall) / (precision + recall)
        return 2.0 * precision * recall / (precision + recall + 1e-7)

    def reset_states(self):
        # Clear out the variables for a new epoch or new evaluation
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


### Autoencoder Model MNIST
#with @tf.strategy?
def ssi_acc( y_true, y_pred):
        max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
        return tf.reduce_mean(
            tf.image.ssim(y_true, y_pred, max_val=max_max, filter_size=3, filter_sigma=1.5, k1=0.01, k2=0.03))
class ElegansMnist280Recon:
    def __init__(self, path_connectome= path_to_connectome):
        (self.x_train, _), (self.x_test, _) = tf.keras.datasets.mnist.load_data()

        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        self.x_train =  tf.where(self.x_train  >= 0.3, 1.0, 0.0)
        self.x_train = np.asarray(self.x_train).astype("float")

        self.x_test =  tf.where(self.x_test  >= 0.3, 1.0, 0.0)
        self.x_test = np.asarray(self.x_test).astype("float")

        self.x_train = tf.expand_dims(self.x_train, -1)
        self.x_test  = tf.expand_dims(self.x_test,  -1)

        self.sample_size,      self.hi, self.wi, self.chi = self.x_train.shape
        self.sample_size_test, self.hi, self.wi, self.chi = self.x_test.shape

        self.graph_ith_aug = path_to_connectome


        self.elegans_graph = Connectome_Reader(self.graph_ith_aug)
        self.gg = self.elegans_graph.read(1, 0, 0)
        ## par network tensor
        self.unique_name = 0
        seq_vs = self.gg.vs()['name']
        self.seq_vs = [int(x) for x in seq_vs]

        self.matrix = np.zeros((self.gg.vcount(), self.gg.vcount()))
        self.matrix_original = self.get_mat_original()

    def net_tensor_track(self, i, j, el=None):
        # print("-------------")
        if el == "E":
            self.matrix[i][j] = 1
            self.matrix[j][i] = 1
        else:
            self.matrix[i][j] = 1

        # print(self.matrix)
        # print(np.matrix(self.matrix_original))
        # input()

        # print("Estimated")
        # print(self.matrix)
        # print("Original")

    def get_tensor_track(self, i, j, el=None):
        print("-------------")
        return self.matrix[i][j]

    def t_n(self, label, synapse_type, node_a, node_b):
        self.unique_name += 1
        nn = label + "_" + str(self.unique_name) + str(synapse_type) + "_" + str(node_a) + str(node_b) + ""
        return nn

    def get_mat_original(self):
        g = self.gg
        # inizializza la matrice di adiacenza
        adjacency_matrix = [[0 for x in range(g.vcount())] for y in range(g.vcount())]

        # scorre tutti i vertici
        for i in range(0, g.vcount()):
            for j in range(0, g.vcount()):
                if g.are_connected(i, j):
                    edge = g.get_eid(i, j)
                    if g.es[edge]["synapse_type"] == "E":
                        adjacency_matrix[i][j] = 1
                        adjacency_matrix[j][i] = 1
                        # if g.vs[j]["role"] == 0 and g.vs[i]["role"] == 0:
                        #    adjacency_matrix[i][j] = 7

        for i in range(0, g.vcount()):
            for j in range(0, g.vcount()):
                if g.are_connected(i, j):
                    edge = g.get_eid(i, j)
                    if g.es[edge]["synapse_type"] == "C":
                        adjacency_matrix[i][j] = 1
                        # if g.vs[i]["role"] == 0 and g.vs[j]["role"] == 0:
                        #    adjacency_matrix[i][j] = 3

        return adjacency_matrix

    def init_graph_allocations(self, x, sensors, interneurons, motors, sensors_to, motors_to, interneurons_to,
                               patch_pattern, activation_line):
        apply_cross_conn = True
        role_list = self.gg.vs()['role']
        i = 0
        for el in role_list:
            if el in ['S', 0]:
                sensors[i] = x  # the input
            i += 1

        print("S to I and M")
        for i in range(0, self.gg.vcount()):
            for j in range(0, self.gg.vcount()):
                try:
                    coord_0 = int(i)
                    coord_1 = int(j)

                    if not isinstance(sensors[coord_0], tf.keras.Sequential):
                        if self.gg.vs()['role'][coord_0] in ["S", 0]:
                            if self.gg.vs()['role'][coord_1] in ["M", 1]:
                                if isinstance(motors[coord_1], tf.keras.Sequential):
                                    gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]

                                    e = tf.TensorSpec.from_spec(sensors[coord_0],
                                                                self.t_n(str(self.gg.vs()['role'][coord_1]),
                                                                         gg_el["synapse_type"],
                                                                         coord_0, coord_1))
                                    rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                    # print("Trovato" + rel_s)
                                    motors[coord_1] = tf.keras.layers.Dense(patch_pattern,
                                                                            input_shape=e.shape, dtype=e.dtype,
                                                                            activation=activation_line,
                                                                            name=self.t_n(rel_s, gg_el["synapse_type"],
                                                                                          coord_0, coord_1))(
                                        sensors[coord_0])
                                    self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])

                                else:
                                    if apply_cross_conn:
                                        gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                        rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                        # print("Trovato" + rel_s)
                                        motors[coord_1] = tf.keras.layers.Multiply(
                                            name=self.t_n(rel_s, gg_el["synapse_type"],
                                                          coord_0, coord_1))([sensors[coord_0], motors[coord_1]])
                                        self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])

                            if self.gg.vs()['role'][coord_1] in ["NA", 2]:
                                if isinstance(interneurons[coord_1], tf.keras.Sequential):
                                    gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                    e = tf.TensorSpec.from_spec(sensors[coord_0],
                                                                self.t_n(str(self.gg.vs()['role'][coord_1]),
                                                                         gg_el["synapse_type"],
                                                                         coord_0, coord_1))
                                    rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                    # print("Trovato" + rel_s)
                                    interneurons[coord_1] = tf.keras.layers.Dense(patch_pattern,
                                                                                  input_shape=e.shape, dtype=e.dtype,
                                                                                  activation=activation_line,
                                                                                  name=self.t_n(rel_s,
                                                                                                gg_el["synapse_type"],
                                                                                                coord_0, coord_1))(
                                        sensors[coord_0])
                                    self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])
                                else:
                                    if apply_cross_conn:
                                        gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                        rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                        # print("Trovato" + rel_s)
                                        interneurons[coord_1] = tf.keras.layers.Multiply(
                                            name=self.t_n(rel_s, gg_el["synapse_type"],
                                                          coord_0, coord_1))([sensors[coord_0], interneurons[coord_1]])
                                        self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])

                                    # input()
                except igraph._igraph.InternalError:
                    continue

        print("NA to S and M")
        for i in range(0, self.gg.vcount()):
            for j in range(0, self.gg.vcount()):
                try:
                    coord_0 = int(i)
                    coord_1 = int(j)
                    if not isinstance(interneurons[coord_0], tf.keras.Sequential):
                        if self.gg.vs()['role'][coord_0] in ["NA", 2]:
                            if self.gg.vs()['role'][coord_1] in ["S", 1]:
                                if isinstance(sensors[coord_1], tf.keras.Sequential):
                                    gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                    e = tf.TensorSpec.from_spec(interneurons[coord_0],
                                                                self.t_n(str(self.gg.vs()['role'][coord_1]),
                                                                         gg_el["synapse_type"],
                                                                         coord_0, coord_1))
                                    rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                    # print("Trovato" + rel_s)
                                    sensors[coord_1] = tf.keras.layers.Dense(patch_pattern,
                                                                             input_shape=e.shape, dtype=e.dtype,
                                                                             activation=activation_line,
                                                                             name=self.t_n(rel_s, gg_el["synapse_type"],
                                                                                           coord_0, coord_1))(
                                        interneurons[coord_0])
                                    self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])
                                else:
                                    if apply_cross_conn:
                                        gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                        rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                        # print("Trovato" + rel_s)
                                        sensors_to[coord_1] = tf.keras.layers.Multiply(
                                            name=self.t_n(rel_s, gg_el["synapse_type"],
                                                          coord_0, coord_1))([interneurons[coord_0], sensors[coord_1]])
                                        self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])

                            if self.gg.vs()['role'][coord_1] in ["M", 1]:
                                if isinstance(motors_to[coord_1], tf.keras.Sequential):
                                    gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                    e = tf.TensorSpec.from_spec(interneurons[coord_0],
                                                                self.t_n(str(self.gg.vs()['role'][coord_1]),
                                                                         gg_el["synapse_type"],
                                                                         coord_0, coord_1))
                                    rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                    # print("Trovato" + rel_s)
                                    motors[coord_1] = tf.keras.layers.Dense(patch_pattern,
                                                                            input_shape=e.shape, dtype=e.dtype,
                                                                            activation=activation_line,
                                                                            name=self.t_n(rel_s,
                                                                                          gg_el["synapse_type"],
                                                                                          coord_0, coord_1))(
                                        interneurons[coord_0])
                                    self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])
                                else:
                                    if apply_cross_conn:
                                        gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                        rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                        # print("Trovato" + rel_s)
                                        motors[coord_1] = tf.keras.layers.Multiply(
                                            name=self.t_n(rel_s, gg_el["synapse_type"],
                                                          coord_0, coord_1))(
                                            [interneurons[coord_0], motors[coord_1]])
                                        self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])
                                    # input()
                except igraph._igraph.InternalError:
                    continue

        print("M to S and NA")
        for i in range(0, self.gg.vcount()):
            for j in range(0, self.gg.vcount()):
                try:
                    coord_0 = int(i)
                    coord_1 = int(j)
                    if not isinstance(motors[coord_0], tf.keras.Sequential):
                        if self.gg.vs()['role'][coord_0] in ["M", 1]:
                            if self.gg.vs()['role'][coord_1] in ["S", 0]:
                                if isinstance(sensors[coord_1], tf.keras.Sequential):
                                    gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                    e = tf.TensorSpec.from_spec(motors[coord_0],
                                                                self.t_n(str(self.gg.vs()['role'][coord_1]),
                                                                         gg_el["synapse_type"],
                                                                         coord_0, coord_1))
                                    rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                    # print("Trovato" + rel_s)
                                    sensors[coord_1] = tf.keras.layers.Dense(patch_pattern,
                                                                             input_shape=e.shape, dtype=e.dtype,
                                                                             activation=activation_line,
                                                                             name=self.t_n(rel_s, gg_el["synapse_type"],
                                                                                           coord_0, coord_1))(
                                        motors[coord_0])
                                    self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])
                                else:
                                    if apply_cross_conn:
                                        gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                        rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                        # print("Trovato" + rel_s)
                                        sensors[coord_1] = tf.keras.layers.Multiply(
                                            name=self.t_n(rel_s, gg_el["synapse_type"],
                                                          coord_0, coord_1))(
                                            [motors[coord_0], sensors[coord_1]])
                                        self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])

                            if self.gg.vs()['role'][coord_1] in ["NA", 2]:
                                if isinstance(interneurons[coord_1], tf.keras.Sequential):
                                    gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                    e = tf.TensorSpec.from_spec(motors[coord_0],
                                                                self.t_n(str(self.gg.vs()['role'][coord_1]),
                                                                         gg_el["synapse_type"],
                                                                         coord_0, coord_1))
                                    rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                    # print("Trovato" + rel_s)
                                    interneurons[coord_1] = tf.keras.layers.Dense(patch_pattern,
                                                                                  input_shape=e.shape, dtype=e.dtype,
                                                                                  activation=activation_line,
                                                                                  name=self.t_n(rel_s,
                                                                                                gg_el["synapse_type"],
                                                                                                coord_0, coord_1))(
                                        motors[coord_0])
                                    self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])
                                else:
                                    if apply_cross_conn:
                                        gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                        rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                        # print("Trovato" + rel_s)
                                        interneurons[coord_1] = tf.keras.layers.Multiply(
                                            name=self.t_n(rel_s, gg_el["synapse_type"],
                                                          coord_0, coord_1))(
                                            [motors[coord_0], interneurons[coord_1]])
                                        self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])

                                    # input()
                except igraph._igraph.InternalError:
                    continue

        # sensors       = [item for item in sensors if not isinstance(item, tf.keras.Sequential)]
        # motors        = [item for item in motors if not isinstance(item, tf.keras.Sequential)]
        # interneurons  = [item for item in interneurons if not isinstance(item, tf.keras.Sequential)]

        # x2 = [item for item in x2 if not isinstance(item, tf.keras.Sequential)]
        # print(x2)
        # input()

        return sensors, motors, interneurons

    def init_graph_inter_allocations(self, sensors, motors, interneurons, sensors_to, motors_to, interneurons_to,
                                     patch_pattern, activation_line):
        print("S to S")

        for i in range(0, self.gg.vcount()):
            for j in range(0, self.gg.vcount()):
                try:
                    coord_0 = int(i)
                    coord_1 = int(j)
                    if not isinstance(sensors[coord_0], tf.keras.Sequential):
                        if self.gg.vs()['role'][coord_0] in ["S", 0]:
                            if self.gg.vs()['role'][coord_1] in ["S", 0]:
                                if not isinstance(motors[coord_1], tf.keras.Sequential):
                                    gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                    # e = tf.TensorSpec.from_spec(sensors[coord_0],
                                    #                            self.t_n(str(self.gg.vs()['role'][coord_1]),
                                    #                                     gg_el["synapse_type"],
                                    #                                     coord_0, coord_1))
                                    rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                    # print("Trovato" + rel_s)
                                    # sensors[coord_1] = tf.keras.layers.Dense(patch_pattern,
                                    #                                              input_shape=e.shape, dtype=e.dtype,
                                    #                                              activation=activation_line,
                                    #                                              name=self.t_n(rel_s,
                                    #                                                            gg_el["synapse_type"],
                                    #                                                            coord_0, coord_1))(sensors[coord_0])

                                    sensors[coord_1] = tf.keras.layers.Multiply(
                                        name=self.t_n(rel_s, gg_el["synapse_type"],
                                                      coord_0, coord_1))([sensors[coord_0], sensors[coord_1]])
                                    self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])

                                    # input()
                except igraph._igraph.InternalError:
                    continue

        print("NA to NA")
        for i in range(0, self.gg.vcount()):
            for j in range(0, self.gg.vcount()):
                try:
                    coord_0 = int(i)
                    coord_1 = int(j)
                    if not isinstance(interneurons[coord_0], tf.keras.Sequential):
                        if self.gg.vs()['role'][coord_0] in ["NA", 2]:
                            if self.gg.vs()['role'][coord_1] in ["NA", 2]:
                                if not isinstance(interneurons[coord_1], tf.keras.Sequential):
                                    gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                    # e = tf.TensorSpec.from_spec(interneurons[coord_0],
                                    #                            self.t_n(str(self.gg.vs()['role'][coord_1]),
                                    #                                     gg_el["synapse_type"],
                                    #                                     coord_0, coord_1))
                                    rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                    # print("Trovato" + rel_s)
                                    # interneurons[coord_1] = tf.keras.layers.Dense(patch_pattern,
                                    #                                              input_shape=e.shape, dtype=e.dtype,
                                    #                                              activation=activation_line,
                                    #                                              name=self.t_n(rel_s,
                                    #                                                            gg_el["synapse_type"],
                                    #                                                            coord_0, coord_1))(interneurons[coord_0])

                                    interneurons[coord_1] = tf.keras.layers.Multiply(
                                        name=self.t_n(rel_s, gg_el["synapse_type"],
                                                      coord_0, coord_1))([interneurons[coord_0], interneurons[coord_1]])
                                    self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])
                except igraph._igraph.InternalError:
                    continue

        print("M to M")
        for i in range(0, self.gg.vcount()):
            for j in range(0, self.gg.vcount()):
                try:
                    coord_0 = int(i)
                    coord_1 = int(j)
                    if not isinstance(motors[coord_0], tf.keras.Sequential):
                        if self.gg.vs()['role'][coord_0] in ["M", 1]:
                            if self.gg.vs()['role'][coord_1] in ["M", 1]:
                                if not isinstance(sensors[coord_1], tf.keras.Sequential):
                                    gg_el = self.gg.es[self.gg.get_eid(coord_0, coord_1)]
                                    # e = tf.TensorSpec.from_spec(motors[coord_0],
                                    #                            self.t_n(str(self.gg.vs()['role'][coord_1]),
                                    #                                     gg_el["synapse_type"],
                                    #                                     coord_0, coord_1))
                                    rel_s = str(self.gg.vs()['role'][coord_0]) + str(self.gg.vs()['role'][coord_1])
                                    # print("Trovato" + rel_s)
                                    # motors[coord_1] = tf.keras.layers.Dense(patch_pattern,
                                    #                                              input_shape=e.shape, dtype=e.dtype,
                                    #                                              activation=activation_line,
                                    #                                              name=self.t_n(rel_s,
                                    #                                                            gg_el["synapse_type"],
                                    #                                                            coord_0, coord_1))(motors[coord_0])
                                    motors[coord_1] = tf.keras.layers.Multiply(
                                        name=self.t_n(rel_s, gg_el["synapse_type"],
                                                      coord_0, coord_1))([motors[coord_0], motors[coord_1]])
                                    self.net_tensor_track(coord_0, coord_1, gg_el["synapse_type"])
                except igraph._igraph.InternalError:
                    continue

        # x2 = [item for item in x2 if not isinstance(item, tf.keras.Sequential)]
        # print(x2)
        # input()
        return sensors, motors, interneurons

    def elegans_latent_graph_total(self, x, patch_pattern, activation_line=tf.nn.gelu):

        node_layer = tf.keras.Sequential()
        sensors = np.repeat(node_layer, self.gg.vcount() + 1)

        node_layer = tf.keras.Sequential()
        sensors_to = np.repeat(node_layer, self.gg.vcount() + 1)

        node_layer = tf.keras.Sequential()
        interneurons = np.repeat(node_layer, self.gg.vcount() + 1)

        node_layer = tf.keras.Sequential()
        interneurons_to = np.repeat(node_layer, self.gg.vcount() + 1)

        node_layer = tf.keras.Sequential()
        motors = np.repeat(node_layer, self.gg.vcount() + 1)

        node_layer = tf.keras.Sequential()
        motors_to = np.repeat(node_layer, self.gg.vcount() + 1)

        sensors, motors, interneurons = self.init_graph_allocations(x, sensors, motors, interneurons, sensors_to,
                                                                    motors_to, interneurons_to, patch_pattern,
                                                                    activation_line)
        sensors, motors, interneurons = self.init_graph_inter_allocations(sensors, motors, interneurons, sensors_to,
                                                                          motors_to, interneurons_to, patch_pattern,
                                                                          activation_line)

        sensors = [item for item in sensors if not isinstance(item, tf.keras.Sequential)]
        motors = [item for item in motors if not isinstance(item, tf.keras.Sequential)]
        interneurons = [item for item in interneurons if not isinstance(item, tf.keras.Sequential)]

        indirect_count = 0
        direct_count = 0
        motor_count = 0
        sensor_count = 0
        interneuron_count = 0
        mot_out = []
        unique_name = 0
        for j in range(len(motors)):
            try:
                if not isinstance(motors[j], tf.keras.Sequential):
                    # motors[j] is a KerasTensor; its _keras_history[0] is the Layer object
                    layer_obj  = motors[j]._keras_history[0]
                    layer_name = layer_obj.name
                    print(f"  [motor #{j}] layer_name = {layer_name}")
                    if "SM" in layer_name:
                        mot_out.append(motors[j])
                    if "NAM" in layer_name:
                        mot_out.append(motors[j])
                    if "MM" in layer_name:
                        mot_out.append(motors[j])
            except (IndexError, KeyError):
                continue

        # 1) count in “sensors”:
        for j in range(len(sensors)):
            try:
                layer_obj  = motors[j]._keras_history[0]
                layer_name = layer_obj.name
                sensor_count += 1
                unique_name += 1
                if "C" in layer_name: 
                    direct_count += 1
                if "E" in layer_name:
                    indirect_count += 1
            except (IndexError, KeyError):
                continue

        # 2) count in “motors”:
        for j in range(len(motors)):
            try:
                layer_obj  = motors[j]._keras_history[0]
                layer_name = layer_obj.name
                motor_count += 1
                unique_name += 1
                if "C" in layer_name:
                    direct_count += 1
                if "E" in layer_name:
                    indirect_count += 1
            except (IndexError, KeyError):
                continue

        # 3) count in “interneurons”:
        for j in range(len(interneurons)):
            try:
                layer_obj  = motors[j]._keras_history[0]
                layer_name = layer_obj.name
                interneuron_count += 1
                unique_name += 1
                if "C" in layer_name:
                    direct_count += 1
                if "E" in layer_name:
                    indirect_count += 1
            except (IndexError, KeyError):
                continue

        print("Sensors")
        print(sensors)

        print("Interneurons")
        print(interneurons)

        print("Motors")
        print(motors)

        print("Stack")
        print(mot_out)

        print(bcolors.OKCYAN + "Graph structured latent space with synapse type: " + bcolors.ENDC)
        print(bcolors.OKCYAN + "Indirect (E) layers  : " + bcolors.ENDC + str(indirect_count))
        print(bcolors.OKCYAN + "Direct   (C) layers  : " + bcolors.ENDC + str(direct_count))
        print(bcolors.OKBLUE + "Sensors layers (S)   : " + bcolors.ENDC + str(sensor_count))
        print(bcolors.OKBLUE + "Intern. layers (NA)  : " + bcolors.ENDC + str(interneuron_count))
        print(bcolors.OKBLUE + "Motor   layers (M)   : " + bcolors.ENDC + str(motor_count))
        print(bcolors.OKGREEN + "Unique name count    : " + bcolors.ENDC + str(unique_name))

        # Replace tf-stack implementation
        merged_m1 =  merged_m1 = tf.keras.layers.Lambda(lambda tensors: tf.stack(tensors, axis=0))(mot_out)
        merged_m1 = Rearrange('a b c -> b a c')(merged_m1)

        # Rearrange
        x_expanded = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, axis=1))(x)

        l_x_mot_out = MultiHeadAttention(num_heads=motor_count, key_dim=32, attention_axes=(1, 2))
        output_tensor, _ = l_x_mot_out(x_expanded, merged_m1, return_attention_scores=True)
        print(output_tensor.shape)
        output_tensor = tf.keras.layers.LayerNormalization()(output_tensor)

        # merged_m1 = tf.keras.layers.Average()(mot_out)
        return output_tensor

    def share_w(self, xp, i, flux, xp_l, xe_l, patch_pattern, act_fun, x1):
        xp[i] = tf.keras.layers.Lambda(lambda x: x[:, i, :, :])(flux)
        xp[i] = tf.keras.layers.Flatten()(xp[i])
        xp_l.append(xp[i])
        xe_l.append(self.elegans_latent_graph_total(xp_l[-1], x1, patch_pattern, act_fun))
        return xp_l, xe_l

    def build6(self, m_lr=0.01, m_epochs=3, batch_size=32, ratio=1, node_number=280, patch_size=14):
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed=141295)
        bias_initializer = tf.zeros_initializer
        act_fun = "elu"
        reg_val3 = 1e-4 #a 4 fa 0.99296 / a 3 col vecchio modello e faceva 98.8
        factor = 15
        patch_pattern = 784
        input_img = tf.keras.Input(shape=(28, 28, 1))

        x = tf.keras.layers.Conv2D(16*factor, (3, 3), activation=act_fun,        padding='same',
                                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, use_bias=True)(input_img)

        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        x = tf.keras.layers.Conv2D(8*factor, (3, 3), activation=act_fun,         padding='same',
                                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, use_bias=True)(x)

        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        encoded_k = tf.keras.layers.Conv2D(8*factor, (3, 3), activation=act_fun, padding='same',
                                           kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, use_bias=True)(x)

        xe_l = []
        xp_l = []
        xp = tf.keras.layers.Flatten()(encoded_k)

        xp = tf.keras.layers.Dense(patch_pattern, activation=act_fun,  #proviamo 1024
                                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, use_bias=True)(xp)

        xp = tf.keras.layers.LayerNormalization(name="S_init")(xp)
        xp_l.append(xp)
        xe_l.append(self.elegans_latent_graph_total(xp_l[-1], patch_pattern, activation_line=act_fun))
        xxp = tf.keras.layers.Lambda(lambda L: tf.stack(L, axis=0))(xp_l)
        xxe = tf.keras.layers.Lambda(lambda L: tf.stack(L, axis=0))(xe_l)



        flux_x = Rearrange('n_p b pat -> b n_p pat', n_p=1)(xxp)
        flux_e = Rearrange('n_p b 1 pat -> b n_p pat', n_p=1)(xxe)
        flux_e = tf.keras.layers.LayerNormalization()(flux_e)
        flux_x = tf.keras.layers.LayerNormalization()(flux_x)
        _, a, b = flux_e.shape

        flux_e    =  tf.keras.layers.Reshape((b, a), input_shape=(a, b))(flux_e)
        flux_x    =  tf.keras.layers.Reshape((b, a), input_shape=(a, b))(flux_x)
        flux_e = Rearrange('b a 1 -> b a' )(flux_e)
        flux_x = Rearrange('b a 1 -> b a' )(flux_x)



        encoded_e   = tf.keras.layers.Flatten()(flux_e)  #from elegans
        encoded_x   = tf.keras.layers.Flatten()(flux_x)  #from input

        encoded     = tf.keras.layers.Multiply()([encoded_e, encoded_x])

        encoded     = tf.keras.layers.Normalization()(encoded)

        init_g = tf.keras.initializers.GlorotUniform(seed=631986)

        encoded = tf.keras.layers.Dense(patch_pattern,  activation=act_fun,
                                        kernel_initializer=init_g, bias_initializer=bias_initializer, use_bias=True)(encoded)  # nessuna attivazione, lineare...

        encoded = tf.keras.layers.Dense(512,  activation=act_fun,
                                        kernel_initializer=init_g, bias_initializer=bias_initializer, use_bias=True)(encoded)  # nessuna attivazione, lineare...
        encoded = tf.keras.layers.Dense(256,  activation=act_fun,
                                        kernel_initializer=init_g, bias_initializer=bias_initializer, use_bias=True)(encoded)  # nessuna attivazione, lineare...
        


        encoded = tf.keras.layers.Dense(128,  activation=act_fun,
                                        kernel_initializer=init_g, bias_initializer=bias_initializer, use_bias=True)(encoded)  # nessuna attivazione, lineare...
        encoded = tf.keras.layers.Reshape([4, 4, 8])(encoded)
        encoded = tf.keras.layers.BatchNormalization()(encoded)

        x = tf.keras.layers.Conv2D(8*factor, (3, 3), activation=act_fun, padding='same',
                                   kernel_initializer=init_g, bias_initializer=bias_initializer, use_bias=True)(encoded)


        x = tf.keras.layers.UpSampling2D((2, 2))(x)


        x = tf.keras.layers.Conv2D(8*factor, (3, 3), activation=act_fun, padding='same',
                                   kernel_initializer=init_g, bias_initializer=bias_initializer, use_bias=True)(x)

        x = tf.keras.layers.UpSampling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(16*factor, (3, 3), activation=act_fun, padding='valid',
                                   kernel_initializer=init_g, bias_initializer=bias_initializer, use_bias=True)(x)

        x = tf.keras.layers.UpSampling2D((2, 2))(x)

        decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same',
                                        kernel_initializer=init_g, bias_initializer=bias_initializer,
                                        kernel_regularizer  =regularizers.L1(reg_val3),
                                        bias_regularizer    =regularizers.L1(reg_val3),
                                        activity_regularizer=regularizers.L1(reg_val3), use_bias=True)(x)


        opt = tf.keras.optimizers.Adam(learning_rate=m_lr)

        met0  = F1Metric(threshold=0.252, name="m0")
        met1  = F1Metric(threshold=0.25,  name="m1")


        model = tf.keras.Model(input_img, [decoded])

        model.compile(optimizer = opt,
                      loss      = 'binary_crossentropy',
                      metrics   = ['accuracy', met0, met1])



        return model

    def build6_train(self, m_lr=0.01, m_epochs=3, batch_size=32, ratio=1, node_number=280, patch_size=28 ):
        checkpoint_filepath = save_to + "/" + subd + "_last_best.keras"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_m1',
            verbose=1,
            save_best_only=True,
            mode='max',
            save_freq='epoch')

        #tb_callback = tf.keras.callbacks.TensorBoard(log_dir= save_to + "/logs")
        #my_callback = check_gradient()


        node_number = int(self.gg.ecount())
        model = self.build6(m_lr, m_epochs, batch_size, ratio, node_number, patch_size)

        csv_callback = tf.keras.callbacks.CSVLogger(save_to + "/logs.csv", separator=',', append=False)
        #tensorboard --logdir=/home/neuronelab/Scrivania/multi-dyad/cc10/logs
        model.fit(self.x_train, self.x_train, shuffle=True, batch_size=batch_size, epochs=m_epochs,
                            validation_data=(self.x_test, self.x_test),  callbacks=[cp_callback, csv_callback])

        results = model.evaluate(self.x_test, self.x_test, batch_size=batch_size, verbose=0)

        model.save(save_to + "/" + subd + "_lb", include_optimizer=True, save_format = 'tf')

        dot_img_file = './graph_plot/model_v1.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_layer_names=True, show_shapes=True)
        dot_obj = tf.keras.utils.model_to_dot(model)
        graph_png = dot_obj.create_png()
        with open('./graph_plot/model_v2.png', 'wb') as f:
            f.write(graph_png)
        return model

    def build6_test(self, model,  m_lr=0.01, m_epochs=3, batch_size=32, ratio=1, node_number=280, patch_size=28, subd="org", save_to="./mnist_autoenc/"):
        weights_path = "./mnist_autoenc/org_last_best.hdf5"
        model.load_weights(weights_path)

        checkpoint_sm = save_to + "/" + subd + "_lb"
        print(checkpoint_sm)
        #dependencies = {
        #    "m0": tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.2),
        #    "m1": tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.25)
        #}
        #model = tf.keras.models.load_model(checkpoint_sm, custom_objects = dependencies, compile=True)
        results = model.evaluate(self.x_test, self.x_test, batch_size=batch_size, verbose=0)
        print("load model eval:::")
        print(results)




with strategy.scope():
    mod = ElegansMnist280Recon().build6_train(learning_rate, num_epochs, batch_size, ratio)
    ElegansMnist280Recon().build6_test(mod, learning_rate, num_epochs, batch_size, ratio, subd, save_to)





exit()


