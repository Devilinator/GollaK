import numpy as np
import keras
import tensorflow.compat.v1 as tf
from keras.layers import Layer
from keras.layers import Input, Conv2D, Dense, Flatten, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import BatchNormalization, Activation, AveragePooling2D
from keras.optimizers import Adam
import keras as K

from resnext import ResNext

activation = 'relu'
last_activation = 'linear'
    
Depth = 20
Cardinality = 8
Width = 16
nb_classes = 2


# final model
def get_eye_tracker_model(img_ch, img_cols, img_rows):
    
    img_dim = (img_ch, img_rows, img_cols)
    
    model = ResNext(img_dim, depth=Depth, cardinality=Cardinality, width=Width, weights=None, classes=nb_classes)
    # model.get_layer(name='resnext').name='resnext_1'
    model.name = 'Left_Right_Eye_Face_net'
    for layer in model.layers:
        layer.name = layer.name + str("_1")
    EYE_NET = ResNext((1024, 16, 32), depth=Depth, cardinality=Cardinality, width=Width, weights=None, classes=nb_classes)
    # EYE_NET.get_layer(name='resnext').name='resnext_2'
    EYE_NET.name = 'Middle_Eye_net'
    for layer in EYE_NET.layers:
        layer.name = layer.name + str("_2")
    model1 = ResNext((1, 25, 25), depth=Depth, cardinality=Cardinality, width=Width, weights=None, classes=nb_classes)
    # model1.get_layer(name='resnext').name='resnext_3'
    model1.name = 'Face_Grid_Net'
    for layer in model1.layers:
        layer.name = layer.name + str("_3")

    # get partial models
    eye_net = model
    face_net_part = model
    face_grid_net = model1

    # right eye model
    right_eye_input = Input(shape=(img_ch, img_cols, img_rows))
    right_eye_net = eye_net(right_eye_input)

    # left eye model
    left_eye_input = Input(shape=(img_ch, img_cols, img_rows))
    left_eye_net = eye_net(left_eye_input)
    
    # EYE_NET or CNN3 In proposal
    e = concatenate([left_eye_net, right_eye_net])
    EYE = EYE_NET(e)

    # face model
    face_input = Input(shape=(img_ch, img_cols, img_rows))
    face_net = face_net_part(face_input)

    # face grid
    face_grid = Input(shape=(1, 25, 25))
    face_grid_net = face_grid_net(face_grid)
    
    # face_c1 = concatenate([face_net, face_grid])
    # print('face_c1',face_c1)

    # dense layers for eyes
    # e = concatenate([left_eye_net, right_eye_net])
    e = Flatten()(EYE)
    fc_e1 = Dense(128, activation=activation)(e)

    # dense layers for face
    f = Flatten()(face_net)
    fc_f1 = Dense(256, activation=activation)(f)
    fc_f2 = Dense(128, activation=activation)(fc_f1)

    # dense layers for face grid
    fg = Flatten()(face_grid_net)
    fc_fg1 = Dense(256, activation=activation)(fg)
    fc_fg2 = Dense(128, activation=activation)(fc_fg1)

    # final dense layers
    h = concatenate([fc_e1, fc_f2, fc_fg2])
    fc1 = Dense(128, activation=activation)(h)
    fc2 = Dense(2, activation=last_activation)(fc1)

    # final model
    final_model = Model(
        inputs=[right_eye_input, left_eye_input, face_input, face_grid],
        outputs=[fc2])

    return final_model
