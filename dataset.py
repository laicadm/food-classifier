import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Dataset:
    
    def __init__(self, dataset_path, batch_size):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
    
    def trainData(self):
        train_data = {}

        train_data['train_data_dir'] = os.path.join(self.dataset_path, 'train')

        train_data['train_datagen'] = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        train_data['train_generator'] = train_data['train_datagen'].flow_from_directory(
            train_data['train_data_dir'],
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode='binary'
        )

        class_indices = train_data['train_generator'].class_indices
        print(class_indices)
        # train_data['class_labels'] = {v: k for k, v in class_indices.items()}

        train_data['train_dataset'] = tf.data.Dataset.from_generator(
            lambda: train_data['train_generator'],
            output_signature=(
                tf.TensorSpec(shape=(None, 150, 150, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        ).repeat()

        return train_data
    
    def validationData(self):
        val_data = {}

        val_data['val_data_dir'] = os.path.join(self.dataset_path, 'validation')

        val_data['val_datagen'] = ImageDataGenerator(rescale=1./255)

        val_data['val_generator'] = val_data['val_datagen'].flow_from_directory(
            val_data['val_data_dir'],
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode='binary'
        )

        val_data['val_dataset'] = tf.data.Dataset.from_generator(
            lambda: val_data['val_generator'],
            output_signature=(
                tf.TensorSpec(shape=(None, 150, 150, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        ).repeat()

        return val_data
    
    def testData(self):
        test_data = {}

        test_data['test_data_dir'] = os.path.join(self.dataset_path, 'test')

        test_data['test_datagen'] = ImageDataGenerator(rescale=1./255)

        test_data['test_generator'] = test_data['test_datagen'].flow_from_directory(
            test_data['test_data_dir'],
            target_size=(150, 150),
            batch_size=self.batch_size,
            class_mode='binary'
        )

        return test_data
    
    

    
    