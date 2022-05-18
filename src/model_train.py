#defining a function with which create a new net using a pre-trained model, or a completly new structure
def neural_struct(VGG_structure=False, fine_tuning=0):
    if VGG_structure:
        #load VGG16 and apply transfer learning
        preTrainedNet = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_size)

        #defining the amount of fine tuning, setting "trainable" propriety for each layer
        for layer in preTrainedNet.layers[:-fine_tuning]:
                layer.trainable = False

        #adding a couple of layers of 1024 neuorns and the output layer
        outModel = preTrainedNet.output
        outModel = GlobalAveragePooling2D()(outModel)
        outModel = Flatten()(outModel)
        outModel = Dense(1024, activation="relu")(outModel)
        outModel = Dropout(0.2)(outModel)
        outModel = Dense(1024, activation="relu")(outModel)
        outModel = Dropout(0.2)(outModel)
        outModel = Dense(num_classes, activation="softmax")(outModel)
        newModel = tf.keras.Model(preTrainedNet.input, outModel)      
    else:
        #create a new neural network
        newModel = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu', input_shape=input_size, padding='same'),
          tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), activation='relu', padding='same'),
          tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'),
          tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Conv2D(512, (3,3), strides=(1,1), activation='relu', padding='same'),
          tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(2048, activation='relu'),
          tf.keras.layers.Dropout(0.4),
          tf.keras.layers.Dense(num_classes, activation='softmax')
       ])

    #compile the model, choosing the optimizer and metric
    newModel.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
        )
    
    return newModel

newModel =  neural_struct(True, 2)
newModel.summary()

#set all parameters of early-stopping
earlyStop = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, restore_best_weights=True)

#train the net with local GPU
with tf.device("/device:GPU:0"):
    historyModel = newModel.fit(train_generator,
              steps_per_epoch = train_steps,
              epochs = num_epochs,
              validation_data = val_generator,
              validation_steps = val_steps,
              callbacks=[earlyStop],
              )

#get accuracy and loss on test
score = newModel.evaluate(test_generator, steps=test_steps, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#print accuracy and loss graph
def plot_loss_accuracy(historyPar):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(historyPar["loss"],'r-x', label="Train Loss")
    ax.plot(historyPar["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(historyPar["accuracy"],'r-x', label="Train accuracy")
    ax.plot(historyPar["val_accuracy"],'b-x', label="Validation accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)

plot_loss_accuracy(historyModel.history)

#save the model and his training history
newModel.save(save_path_name)
pd.DataFrame.from_dict(historyModel.history).to_csv(save_history_path_name, index=False)