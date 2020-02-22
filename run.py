#!/usr/bin/env python
import csv
import os
import random
from optparse import OptionParser

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

__author__ = "Dimi Balaouras"
__copyright__ = "Copyright 2020, uth.gr"
__license__ = "Apache License 2.0, see LICENSE for more details."
__version__ = "0.0.1"
__description__ = ""
__abs_dirpath__ = os.path.dirname(os.path.abspath(__file__))


def load_icd10_file():
    """
    """
    with open(cli_options.icd_file, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile, delimiter=',')
        # extracting each data row one by one
        icd10_codes = [row[1] for row in csvreader]

    return icd10_codes


def create_dataset():
    """
    Create a artificial PHR dataset
    """
    print("Now creating PHR dataset")
    # erase cli_options.features_export_file
    open(cli_options.features_export_file, 'w').close()

    icd10_codes = load_icd10_file()
    age_range = range(1, 100)
    genders = [0, 1]
    random.seed(1)
    for i in range(cli_options.size_of_phr):
        # age = random.choice(age_range)
        gender = random.choice(genders)
        idc10_sampled_codes = random.sample(icd10_codes,
                                            random.randint(cli_options.min_cond_num, cli_options.max_cond_num))
        features_list = [0] * len(icd10_codes)
        for idc10_code in idc10_sampled_codes:
            index = icd10_codes.index(idc10_code)
            features_list[index] = 1

        features_list.insert(0, gender)
        features_list.insert(0, age)

        with open(cli_options.features_export_file, 'a') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
            wr.writerow(features_list)


def create_model(data):
    """

    :return: The Autoencoder model
    """

    print("Now creating model")
    input_dim = data.shape[1]  # 329 features
    encoding_dim = 150

    # 5 connected layers with 329, 150, 75, 75 and 329 neurons respectively
    input_layer = Input(shape=(input_dim,))

    encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
    decoder = Dense(int(encoding_dim), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)

    model = Model(inputs=input_layer, outputs=decoder)

    # loss function and optimizer
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, train_data, validation_data, noise=None):
    """
    Train the model
    """
    print("Now training model")

    nb_epoch = 100
    batch_size = 32

    checkpointer = ModelCheckpoint(filepath=cli_options.checkpoint_path, verbose=0, save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    train_data_dis = train_data

    for row in train_data:
        print(row)

    history = model.fit(train_data_dis, train_data,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=validation_data,
                        verbose=1,
                        callbacks=[checkpointer, tensorboard]).history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')

    plt.show()


def restore_model(model):
    """
    Restore the model from stored weights
    TODO: error checking
    """
    model.load_weights(cli_options.checkpoint_path)


def run():
    """
    Runs the various operations
    """
    if cli_options.generate:
        create_dataset()

    # import data
    data = pd.read_csv(cli_options.features_export_file)
    pd.set_option('display.max_rows', data.shape[0] + 1)


    # split train and test datasets, keep 20% for testing purposes
    x_train, x_test, = train_test_split(data, test_size=0.2, random_state=42)

    model = create_model(data)

    if cli_options.restore_encoder:
        restore_model(model)

    if cli_options.train_encoder:
        train_model(model, x_train, (x_test, x_test), noise=None)

    x_test1, x_test2, = train_test_split(x_train, test_size=0.05, random_state=42)

    encoded_data = model.predict(x_test1)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(x_test1)
    #     print(encoded_data)

    loss, acc = model.evaluate(x_test1, x_test1, verbose=2)
    print("Model, accuracy: {:5.2f}%".format(100 * acc))


def parse_options():
    """
    Parses all options passed via the command line.
    """

    # Define version and usage strings
    usage = "python %prog [options]"

    # Initiate the cli options parser
    parser = OptionParser(usage=usage, version=__version__, description=__description__)

    # Define available command line options

    parser.add_option("-g", "--generate", action="store_true", dest="generate", default=False,
                      help="Generate PHR data")

    parser.add_option("-t", "--train", action="store_true", dest="train_encoder", default=False,
                      help="Train Autoencoder")

    parser.add_option("-r", "--restore", action="store_true", dest="restore_encoder", default=False,
                      help="Restore Autoencoder")

    parser.add_option("-c", "--checkpoint_path", action="store", dest="checkpoint_path", default="model.h5",
                      help="Path to store model weights")

    parser.add_option("-d", "--icd_file", action="store", dest="icd_file", default="icd10.csv",
                      help="Path to icd10 data csv file")

    parser.add_option("-f", "--features_export_file", action="store", dest="features_export_file",
                      default="features_export_file.csv",
                      help="Filename of the exported random feature file")

    parser.add_option("-s", "--size_of_phr", action="store", dest="size_of_phr", default=1000,
                      type="int", help="Number of PHRs to generate")

    parser.add_option("--min_cond_num", action="store", dest="min_cond_num", default=3,
                      type="int", help="Minimum number of conditions per EHR")

    parser.add_option("--max_cond_num", action="store", dest="max_cond_num", default=5,
                      type="int", help="Maximum number of conditions per EHR")

    # Parse the options
    (options, args) = parser.parse_args()

    return options, args


if __name__ == '__main__':
    """
    Entry point of this module. This method will be invoked first when running the app.
    """

    # Parse command line options
    (cli_options, cli_args) = parse_options()

    print("\n")

    # Run the app
    run()
