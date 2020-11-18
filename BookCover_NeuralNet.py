# import the necessary packages
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import urllib.request


from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import concatenate
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model


class NN_models:

    def create_mlp(Self, dim, regress=False):
        # define our MLP network
        model = Sequential()
        model.add(Dense(8, input_dim=dim, activation="relu"))
        model.add(Dense(4, activation="relu"))

        # check to see if the regression node should be added
        if regress:
            model.add(Dense(1, activation="linear"))

        # return our model
        return model

    def create_cnn(Self, height, width, depth, filters=(16, 32, 64), regress=False):
        # initialize the input shape and channel dimension, assuming
        # TensorFlow/channels-last ordering

        filters = np.asarray(filters)
        input_shape = (height, width, depth)
        chanDim = -1

        # define the model input
        inputs = Input(shape=input_shape)

        # loop over the number of filters
        for i in range(filters.shape[0]):
            # if this is the first CONV layer then set the input
            # appropriately
            f = filters[i]
            if i == 0:
                x = inputs

            # CONV => RELU => BN => POOL
            x = Conv2D(f, (3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # flatten the volume, then FC => RELU => BN => DROPOUT
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)

        # apply another FC layer, this one to match the number of nodes
        # coming out of the MLP
        x = Dense(4)(x)
        x = Activation("relu")(x)

        # check to see if the regression node should be added
        if regress:
            x = Dense(1, activation="linear")(x)

        # construct the CNN
        model = Model(inputs, x)

        # return the CNN
        return model


def download_images(data, dest):

    for i in data.newbookid:
        file_path_s = dest + "\\" + str(i) + '.jpg'
        urllib.request.urlretrieve(data.small_image_url[i], file_path_s)



def load_cover_images(df, input_path):
    # initialize our images array
    images = []

    # loop over the indexes of the books
    for i in df.newbookid:

        path = input_path + "\\" + str(i) + '.jpg'
        print(path)
        image = cv2.imread(path)
        outputImage = cv2.resize(image, (32, 32))
        images.append(outputImage)

    # return our set of images
    return np.array(images)


def Create_cover_image_data(train_data, test_data, images, scaling):
    # process and filter data and label per for the CNN

    train_Y = train_data.average_rating
    test_Y = test_data.average_rating

    train_images = []
    for j in train_data.newbookid:
        train_images.append(images[j - 1])
    test_images = []
    for j in test_data.newbookid:
        test_images.append(images[j - 1])

    train_Y = np.asarray(train_Y) / scaling
    test_Y = np.asarray(test_Y) / scaling

    return np.asarray(train_images), train_Y, np.asarray(test_images), test_Y


def Create_user_book_data(train_data, test_data, data, scaling):
    # process and filter data and label per for the NN

    train_Y = train_data.average_rating
    test_Y = test_data.average_rating

    # Continous data

    cont = ["original_publication_year", "pages"]
    cs = MinMaxScaler()

    trainCont = cs.fit_transform(train_data[cont])
    testCont = cs.transform(test_data[cont])

    # Categorical data
    categ = ["first_author", "title", "firstgenre"]

    for j in range(len(categ)):
        bin = LabelBinarizer().fit(data[categ[j]])
        if j == 0:
            trainFull = np.hstack([trainCont, bin.transform(train_data[categ[j]])])
            testFull = np.hstack([testCont, bin.transform(test_data[categ[j]])])
        else:
            trainFull = np.hstack([trainFull, bin.transform(train_data[categ[j]])])
            testFull = np.hstack([testFull, bin.transform(test_data[categ[j]])])

    train_Y = np.asarray(train_Y) / scaling
    test_Y = np.asarray(test_Y) / scaling

    return trainFull, train_Y, testFull, test_Y


def split_data(data, split, n):
    allbooks = random.sample(list(data.newbookid), n)
    split_train = np.around(n * split)
    train = data[data['newbookid'].isin(allbooks[0:int(split_train) - 1])]
    test = data[data['newbookid'].isin(allbooks[int(split_train):n])]

    return test, train


def NN_fit(train_data, train_Y, test_data, test_Y, model):
    # train the model
    print("[INFO] training model...")
    m = model.fit(train_data, train_Y, validation_data=(test_data, test_Y), epochs=25, batch_size=8)

    # make predictions on the testing data
    print("[INFO] predicting book ratings...")
    preds_test = model.predict(test_data)
    preds_train = model.predict(train_data)

    # compute the difference between the *predicted* book rating and the
    # *actual* rating, then compute the percentage difference and
    # the absolute percentage difference
    diff = preds_test.flatten() - test_Y
    percentDiff = (diff / test_Y) * 100
    absPercentDiff = np.abs(percentDiff)

    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(absPercentDiff)
    std = np.std(absPercentDiff)

    return preds_test, preds_train, mean, std, m.history['val_loss']


## Main

root_path = r'C:\Backup\ayoffe\Desktop\Stanford\Machine Learning\Project\goodbooks-10k-master' # define root folder where dsta csv is located

if not os.path.isdir(root_path + r'\NN'):
    os.mkdir(root_path + r'\NN')
save_path = root_path + r'\NN'
data = pd.read_csv(root_path + r'\finalbooks.csv')

# Download images from web

if not os.path.isdir(root_path + r'\img_s'):
    os.mkdir(root_path + r'\img_s')
input_path = root_path + r'\img_s'
# download_images(data, input_path)

data_orig = data
data = data.dropna()
# Concert pages string to number
for i in data.newbookid:
    temp = data.pages[data.newbookid == i].astype(str)
    temp = temp.str.split(' ')
    data.pages[data.newbookid == i] = int(temp.str[0])




# load images

images = load_cover_images(data_orig, input_path)
images = images / 255.0

NN_type = np.asarray([1, 2, 3])  # 1 - CNN based on cover images, 2 - NN based on dataset, 3 - combined NN
scaling = 5  # rating normalization

n_train = np.asarray([20, 1200, 500, 1000, 2000, 4000, data.shape[0]])

RMS = np.zeros((n_train.shape[0], 4))  # cols: number of books; image based RMSE, data based RMSE, mix based RMSE
c = -1
for u in n_train:
    c += 1
    RMS[c, 0] = u

    test_data, train_data = split_data(data, 0.75, u)

    for n_type in NN_type:
        preds = []
        loss_vec = []
        train_vec = []
        # create model
        model = NN_models()

        if n_type == 1:
            model_CNN = model.create_cnn(32, 32, 3, regress=True)
            opt = Adam(lr=1e-3, decay=1e-3 / 200)
            model_CNN.compile(loss="mean_absolute_percentage_error", optimizer=opt)
            [train_images, train_Y, test_images, test_Y] = Create_cover_image_data(train_data, test_data, images,
                                                                                   scaling)
            [preds_test, preds_train, mean, std, loss] = NN_fit(train_images, train_Y, test_images, test_Y, model_CNN)
            np.savetxt(save_path + r'\Preds_Test_CNN_' + str(u) + '.csv', preds_test)
            np.savetxt(save_path + r'\Preds_Train_CNN_' + str(u) + '.csv', preds_train)
            np.savetxt(save_path + r'\Loss_CNN_' + str(u) + '.csv', loss)
            rmse_test = np.sqrt(np.mean((preds_test * scaling - np.asarray(test_Y.reshape(-1, 1)) * scaling) ** 2))
            rmse_train = np.sqrt(np.mean((preds_train * scaling - np.asarray(train_Y.reshape(-1, 1)) * scaling) ** 2))
            with open(save_path + r'\RMSE_Test_CNN_' + str(u) + '.txt', 'w') as f:
                f.write('%f' % rmse_test)
            RMS[c, n_type] = rmse_test
            with open(save_path + r'\RMSE_Train_CNN_' + str(u) + '.txt', 'w') as f:
                f.write('%f' % rmse_train)
            RMS[c, n_type] = rmse_test
        elif n_type == 2:
            [trainFull, train_Y, testFull, test_Y] = Create_user_book_data(train_data, test_data, data,
                                                                           scaling)
            model_NN = model.create_mlp(trainFull.shape[1], regress=True)
            opt = Adam(lr=1e-3, decay=1e-3 / 200)
            model_NN.compile(loss="mean_absolute_percentage_error", optimizer=opt)
            [preds_test, preds_train, mean, std, loss] = NN_fit(trainFull, train_Y, testFull, test_Y, model_NN)
            np.savetxt(save_path + r'\Preds_test_MLP_NN_' + str(u) + '.csv', preds_test)
            np.savetxt(save_path + r'\Preds_train_MLP_NN_' + str(u) + '.csv', preds_train)
            np.savetxt(save_path + r'\Loss_MLP_NN_' + str(u) + '.csv', loss)
            rmse_test = np.sqrt(np.mean((preds_test * scaling - np.asarray(test_Y.reshape(-1, 1)) * scaling) ** 2))
            rmse_train = np.sqrt(np.mean((preds_train * scaling - np.asarray(train_Y.reshape(-1, 1)) * scaling) ** 2))
            with open(save_path + r'\RMSE_Test_MLP_NN_' + str(u) + '.txt', 'w') as f:
                f.write('%f' % rmse_test)
            RMS[c, n_type] = rmse_test
            with open(save_path + r'\RMSE_Train_MLP_NN_' + str(u) + '.txt', 'w') as f:
                f.write('%f' % rmse_train)
            RMS[c, n_type] = rmse_test
        else:

            # create the MLP and CNN models

            mlp = model.create_mlp(trainFull.shape[1], regress=False)
            cnn = model.create_cnn(32, 32, 3, regress=False)

            # create the input to our final set of layers as the *output* of both
            # the MLP and CNN
            combinedInput = concatenate([mlp.output, cnn.output])
            # our final FC layer head will have two dense layers, the final one
            # being our regression head
            x = Dense(4, activation="relu")(combinedInput)
            x = Dense(1, activation="linear")(x)

            # our final model will accept categorical/numerical data on the MLP
            # input and images on the CNN input, outputting a single value (the
            # predicted price of the house)
            model = Model(inputs=[mlp.input, cnn.input], outputs=x)
            opt = Adam(lr=1e-3, decay=1e-3 / 200)
            model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

            # train the model
            print("[INFO] training model...")
            m = model.fit(
                [trainFull, train_images], train_Y,
                validation_data=([testFull, test_images], test_Y),
                epochs=25, batch_size=8)

            # make predictions on the testing data
            print("[INFO] predicting book ratings...")
            preds_test = model.predict([testFull, test_images])
            preds_train = model.predict([trainFull, train_images])

            diff = preds_test.flatten() - test_Y
            percentDiff = (diff / test_Y) * 100
            absPercentDiff = np.abs(percentDiff)

            # compute the mean and standard deviation of the absolute percentage
            # difference
            mean = np.mean(absPercentDiff)
            std = np.std(absPercentDiff)
            loss = m.history['val_loss']

            np.savetxt(save_path + r'\Preds_test_Mix_' + str(u) + '.csv', preds_test)
            np.savetxt(save_path + r'\Preds_train_Mix_' + str(u) + '.csv', preds_train)
            np.savetxt(save_path + r'\Loss_Mix_' + str(u) + '.csv', loss)
            rmse_test = np.sqrt(np.mean((preds_test * scaling - np.asarray(test_Y.reshape(-1, 1)) * scaling) ** 2))
            rmse_train = np.sqrt(np.mean((preds_train * scaling - np.asarray(train_Y.reshape(-1, 1)) * scaling) ** 2))
            with open(save_path + r'\RMSE_Test_MIX_' + str(u) + '.txt', 'w') as f:
                f.write('%f' % rmse_test)
            RMS[c, n_type] = rmse_test
            with open(save_path + r'\RMSE_Train_MIX_' + str(u) + '.txt', 'w') as f:
                f.write('%f' % rmse_train)
            RMS[c, n_type] = rmse_test

fullRMSE = pd.DataFrame(
    {'n_users': RMS[:, 0], 'Image_based': RMS[:, 1], 'Data_based': RMS[:, 2], 'Mix_Model': RMS[:, 3]})
fullRMSE.to_csv(save_path + r'\All_RMSE.csv', index=False)

plt.figure()
fmt_CNN = '[o][-][b]'
fmt_MLP = '[s][-][r]'
fmt_Mix = '[*][-][g]'
plt.plot(RMS[:, 0] * 0.75, RMS[:, 1], 'bo-', label='Cover image based CNN', linewidth=1.5)
plt.plot(RMS[:, 0] * 0.75, RMS[:, 2], 'gs-', label='DataBase based MPL', linewidth=1.5)
plt.plot(RMS[:, 0] * 0.75, RMS[:, 3], 'r^-', label='Mixed Model', linewidth=1.5)
plt.ylim(0, 1)

# Add labels and save to disk
plt.title('RMSE as function of Data size')
plt.xlabel('# of Books for Train set')
plt.ylabel('RMSE on test set')
plt.legend()
plt.savefig(save_path + r'\RMSE_vs_users.tiff')
