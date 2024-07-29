class Model:

    def __init__(self, model, compilers):
        self.model = model
        self.compilers = compilers

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