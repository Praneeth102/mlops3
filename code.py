{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#importing data from keras\n",
    "\n",
    "from keras.datasets import mnist\n",
    "from keras.utils import np_utils\n",
    "import keras\n",
    "from keras.datasets import mnist\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Dropout, Flatten\n",
    "from keras.layers import Conv2D, MaxPooling2D, BatchNormalization\n",
    "from keras import backend as K\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training Parameters\n",
    "batch_size = 128\n",
    "epochs =1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Lets store the number of rows and columns\n",
    "img_rows =28\n",
    "img_cols =28"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    " # the data, split between train and test sets\n",
    "(x_train, y_train), (x_test, y_test) = mnist.load_data('mymnist.data')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Getting our date in the right 'shape' needed for Keras\n",
    "# We need to add a 4th dimenion to our date thereby changing our\n",
    "# Our original image shape of (60000,28,28) to (60000,28,28,1)\n",
    "x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)\n",
    "x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)\n",
    "# store the shape of a single image \n",
    "input_shape = (img_rows, img_cols, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# change our image type to float32 data type\n",
    "x_train = x_train.astype('float32')\n",
    "x_test = x_test.astype('float32')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Normalize our data by changing the range from (0 to 255) to (0 to 1)\n",
    "x_train /= 255\n",
    "x_test /= 255"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('x_train shape:', x_train.shape)\n",
    "print(x_train.shape[0], 'train samples')\n",
    "print(x_test.shape[0], 'test samples')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Now we one hot encode outputs\n",
    "y_train = np_utils.to_categorical(y_train)\n",
    "y_test = np_utils.to_categorical(y_test)\n",
    "\n",
    "# Let's count the number columns in our hot encoded matrix \n",
    "print (\"Number of Classes: \" + str(y_test.shape[1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create model\n",
    "num_classes=10\n",
    "model = Sequential()\n",
    "\n",
    "model.add(Conv2D(32, kernel_size=(3, 3),\n",
    "                 activation='relu',\n",
    "                 input_shape=input_shape))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "model.add(Conv2D(64, (3, 3), activation='relu'))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "model.add(Dropout(0.25))\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(128, activation='relu'))\n",
    "model.add(BatchNormalization())\n",
    "\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(num_classes, activation='softmax'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss = 'categorical_crossentropy',\n",
    "              optimizer = keras.optimizers.Adadelta(),\n",
    "              metrics = ['accuracy'])\n",
    "\n",
    "print(model.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "history = model.fit(x_train, y_train,\n",
    "          batch_size=batch_size,\n",
    "          epochs=epochs,\n",
    "          verbose=1,\n",
    "          validation_data=(x_test, y_test))\n",
    "\n",
    "score = model.evaluate(x_test, y_test, verbose=0)\n",
    "print('Test loss:', score[0])\n",
    "print('Test accuracy:', score[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    " model.save(\"model.h5\")\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pran=score[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pran"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
