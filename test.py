from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# load the model
model = load_model('food_classification_model.keras')

# class indices as basis
class_labels = {
    0: 'Apple Pie',
    1: 'Bibimbap',
    2: 'Caesar Salad',
    3: 'Donuts'
}

# preprocess image
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

try:
    image_path = str(input("Enter image file name: "))
    img_array = preprocess_image('test_img/'+image_path+'.jpg')

    prediction = model.predict(img_array)

    predicted_class_index = np.argmax(prediction, axis=1)[0]

    predicted_class_label = class_labels.get(predicted_class_index, "Unknown class")
    predicted_probability = prediction[0][predicted_class_index]

    print(f'The image is classified as: {predicted_class_label}')
    print(f'Predicted Probability: {round(float(predicted_probability), 5)}')

except Exception as e:
    print(e)