import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten

class Model:

    def __init__(self, model, layers, compilers):
        self.model = model
        self.layers = layers
        self.compilers = compilers

    # dense layers if ever you decide to use dense as a layer
    def addDenseLayer(self):
        self.model.add(Flatten(input_shape=(150, 150, 3)))
        for layer in self.layers:
            try:
                self.model.add(Dense(layer['units'], activation=layer['activation']))
            except Exception as e:
                print(f"Error adding layer: {e}")
    
    def compileModel(self):
        self.model.compile(optimizer=self.compilers['optimizer'], 
                               loss=self.compilers['loss'], 
                               metrics=self.compilers['metrics'])
        
    def fitModel(self, train_generator, steps_per_epoch, 
                 validation_data, validation_steps, epochs):
        self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            validation_steps=validation_steps,
            epochs=epochs
        )

    def summarizeModel(self):
        self.model.summary()

    def getModel(self):
        return self.model