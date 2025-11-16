import keras

def getEdgeDCNet4(input_shape=(160,160,3), activation='sigmoid', loss='binary_crossentropy', optimizer='adam'):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape, name='input'))

    model.add(keras.layers.Conv2D(3,(3,3),padding='valid',use_bias=False,strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D())
    model.add(keras.layers.DepthwiseConv2D((3,3),padding='valid',strides=(2,2),use_bias=False))
    model.add(keras.layers.ReLU(6.))
    model.add(keras.layers.Conv2D(6,(1,1),padding='same',strides=(1,1),use_bias=False))
    model.add(keras.layers.ReLU(6.))

    model.add(keras.layers.ZeroPadding2D())
    model.add(keras.layers.DepthwiseConv2D((3,3),padding='valid',strides=(2,2),use_bias=False))
    model.add(keras.layers.ReLU(6.))
    model.add(keras.layers.Conv2D(12,(1,1),padding='same',strides=(1,1),use_bias=False))
    model.add(keras.layers.ReLU(6.))

    model.add(keras.layers.ZeroPadding2D())
    model.add(keras.layers.DepthwiseConv2D((3,3),padding='valid',strides=(2,2),use_bias=False))
    model.add(keras.layers.ReLU(6.))
    model.add(keras.layers.Conv2D(24,(1,1),padding='same',strides=(1,1),use_bias=False))
    model.add(keras.layers.ReLU(6.))

    model.add(keras.layers.ZeroPadding2D())
    model.add(keras.layers.DepthwiseConv2D((3,3),padding='valid',strides=(2,2),use_bias=False))
    model.add(keras.layers.ReLU(6.))
    model.add(keras.layers.Conv2D(24,(1,1),padding='same',strides=(1,1),use_bias=False))
    model.add(keras.layers.ReLU(6.))

    model.add(keras.layers.MaxPool2D((3,3)))
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2, activation=activation))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

    #model.summary()
    return model
