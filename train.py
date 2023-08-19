import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  #disable for tensorFlow V2
   
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)

#     except RuntimeError as e:
#         print(e)

gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))

# tf.config.gpu.set_per_process_memory_fraction(0.5)
# tf.config.gpu.set_per_process_memory_growth(True)

from keras.optimizers import SGD, Adam
from keras.callbacks import  EarlyStopping, ModelCheckpoint
from load_data import load_data_from_npz, load_batch, load_data_names, load_batch_from_names_random
from models import get_eye_tracker_model
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard


# def lr_schedule(epoch):
#     """Learning Rate Schedule

#     Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
#     Called automatically every epoch as part of callbacks during training.

#     # Arguments
#         epoch (int): The number of epochs

#     # Returns
#         lr (float32): learning rate
#     """
#     LR = 1e-4
#     if epoch > 55:
#         LR = 1e-6
#     elif epoch > 40:
#         LR = 1e-5
#     elif epoch > 26:
#         LR = 1e-4
#     elif epoch > 10:
#         LR = 1e-5
#     print('Learning rate: ', LR)
#     return LR

# generator for data loaded from the npz file
def generator_npz(data, batch_size, img_ch, img_cols, img_rows):

    while True:
        for it in list(range(0, data[0].shape[0], batch_size)):
            x, y = load_batch([l[it:it + batch_size] for l in data], img_ch, img_cols, img_rows)
            yield x, y


# generator with random batch load (train)
def generator_train_data(names, path, batch_size, img_ch, img_cols, img_rows):

    while True:
        x, y = load_batch_from_names_random(names, path, batch_size, img_ch, img_cols, img_rows)
        yield x, y


# generator with random batch load (validation)
def generator_val_data(names, path, batch_size, img_ch, img_cols, img_rows):

    while True:
        x, y = load_batch_from_names_random(names, path, batch_size, img_ch, img_cols, img_rows)
        yield x, y

def train(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev

    #todo: manage parameters in main
    if args.data == "big":
        dataset_path = "E:\gig-dl.ir\Dataset"
        weights_path = "H:\ResNext1 Weights/weights.010-3.34434.hdf5"
        # dataset_path = "G:\Dataset\GazeCapture"
    if args.data == "small":
        dataset_path = "D:\Dataset\eye_tracker_train_and_val.npz"
        weights_path = "C:\GazeEstimation\SE-ResNext\ResNext_weights/weights.004-5.62960.hdf5"
        print("Weights: {}".format(weights_path))

    if args.data == "big":
        # train_path = "F:\GazeCapture/train"
        # val_path = "F:\GazeCapture/validation"
        # test_path = "F:\GazeCapture/test"
        train_path = "E:\gig-dl.ir\Dataset/SETrain"
        val_path = "E:\gig-dl.ir\Dataset/SEValidation"
        test_path = "E:\gig-dl.ir\Dataset/SETest"

    print("{} dataset: {}".format(args.data, dataset_path))

    # train parameters
    n_epoch = args.max_epoch
    batch_size = args.batch_size
    patience = args.patience

    # image parameter
    img_cols = 64
    img_rows = 64
    img_ch = 3

    # model
    model = get_eye_tracker_model(img_ch, img_cols, img_rows)

    # model summary
    model.summary()

    # weights
    print("Loading weights...",  end='')
    weights_path = "ResNext_weights/weights.001-0.80546.hdf5"
    model.load_weights(weights_path)
    print("Done.")

    # lr_scheduler = LearningRateScheduler(lr_schedule)
    # optimizer
    # sgd = SGD(lr=1e-4, decay=5e-4, momentum=9e-1, nesterov=True)
    sgd = SGD(lr=1e-4)
    adam = Adam(lr=0.5e-5)

    # compile model
    model.compile(optimizer=adam, loss='mse')

#    tensorboard = TensorBoard(log_dir='./Graph_V1_20', histogram_freq=1,
#                          write_graph=True, write_images=True,
#                          write_grads=True)

    # data
    # todo: parameters not hardocoded
    if args.data == "big":
        # train data
        train_names = load_data_names(train_path)
        # validation data
        val_names = load_data_names(val_path)
        # test data
        test_names = load_data_names(test_path)

    if args.data == "small":
        train_data, val_data = load_data_from_npz(dataset_path)
#        print("Loading weights...")
#        model.load_weights(weights_path)

    # debug
    # x, y = load_batch([l[0:batch_size] for l in train_data], img_ch, img_cols, img_rows)
    # x, y = load_batch_from_names(train_names[0:batch_size], dataset_path, img_ch, img_cols, img_rows)

    # last dataset checks
    if args.data == "small":
        print("train data sources of size: {} {} {} {} {}".format(
            train_data[0].shape[0], train_data[1].shape[0], train_data[2].shape[0],
            train_data[3].shape[0], train_data[4].shape[0]))
        print("validation data sources of size: {} {} {} {} {}".format(
            val_data[0].shape[0], val_data[1].shape[0], val_data[2].shape[0],
            val_data[3].shape[0], val_data[4].shape[0]))

    if args.data == "big":
        model.fit_generator(
            generator=generator_train_data(train_names, dataset_path, batch_size, img_ch, img_cols, img_rows),
            steps_per_epoch=(len(train_names)) / batch_size,
            epochs=n_epoch,
            verbose=1,
            validation_data=generator_val_data(val_names, dataset_path, batch_size, img_ch, img_cols, img_rows),
            validation_steps=(len(val_names)) / batch_size,
            callbacks=[EarlyStopping(patience=patience),
                       ModelCheckpoint("ResNext_weights/weights.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=False),
                       TensorBoard(log_dir='./Graph_V1_20', histogram_freq=0,
                          write_graph=True, write_images=True)
                       ]
        )
    if args.data == "small":
        model.fit_generator(
            generator=generator_npz(train_data, batch_size, img_ch, img_cols, img_rows),
            steps_per_epoch=(train_data[0].shape[0])/batch_size,
            epochs=n_epoch,
            verbose=1,
            validation_data=generator_npz(val_data, batch_size, img_ch, img_cols, img_rows),
            validation_steps=(val_data[0].shape[0])/batch_size,
            callbacks=[EarlyStopping(patience=patience),
                       ModelCheckpoint("ResNext_weights/weights.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=False),
                       TensorBoard(log_dir='./Graph_V1_20', histogram_freq=0,
                          write_graph=True, write_images=True)
                       ]
        )
