# import
import os
from dotenv import load_dotenv
from models import Model
from dataset import Dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# load env
load_dotenv()

# load dataset: training, validation, testing
init_dataset = Dataset(os.getenv('DATASET_PATH'), 16)
train_data = init_dataset.trainData()
val_data = init_dataset.validationData()
test_data = init_dataset.testData()

# define model params
dense_layer = [
    {'units': 128, 'activation': 'relu'},
    {'units': 64, 'activation': 'relu'}, 
    {'units': 1, 'activation': 'sigmoid'} # activation=softmax for categorical
]
compiler = {
    'optimizer': 'adam',
    'loss': 'binary_crossentropy', #loss=categorical_crossentropy for categorical
    'metrics':  ['accuracy']
}

# get generators
train_generator = train_data['train_generator']
val_generator = val_data['val_generator']
test_generator = test_data['test_generator']

steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
validation_steps = max(1, val_generator.samples // val_generator.batch_size)

# building the model
base_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model = Model(base_model, compiler)
model.compileModel()
model.summarizeModel()
model.fitModel(
    train_generator,
    steps_per_epoch,
    val_generator,
    validation_steps,
    10
)
classifier_model = model.getModel()


# evaluate the model on the test set
test_loss, test_accuracy = classifier_model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Model Accuracy: {test_accuracy * 100}%')

# save model
if test_accuracy * 100 > 50:
    classifier_model.save('food_classification_model.keras')
    print("The model has been saved.")
else:
    print("The model was not saved due to poor performance.")