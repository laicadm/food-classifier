from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# load the model
model = load_model('food_classification_model.keras')

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

    predicted_probability = prediction[0][0]
    predicted_class = (predicted_probability > 0.5).astype(int)

    if predicted_class == 1:
        print('The image is not classified as food.')
    else:
        print('The image is classified as food.')

    # print(f'Predicted Probability: {round(predicted_probability * 100, 2)}%')

except Exception as e:
    print(e)