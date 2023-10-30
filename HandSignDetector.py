import cv2
import numpy as np
from tensorflow import keras

class HandSignDetector:
    def __init__(self, model_path):
        # Load the pre-trained model
        self.model = keras.models.load_model(model_path)



    def preprocess_image(self, image_path):
        # Load and preprocess the input image
        input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        input_image = cv2.resize(input_image, (28, 28))
        input_image = input_image / 255.0
        input_image = input_image.reshape(1, 28, 28, 1)
        return input_image
   
    def preprocess_image2(self, image_path):
        # Load and preprocess the input image
        input_image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (120, 160))
        input_image = input_image / 255.0
        input_image = input_image.reshape(1, 120, 160, 3)
        return input_image
    
    def detect_sign(self, image_path):


        # Preprocess the image
        input_image = self.preprocess_image(image_path)
        # Make predictions using the loaded model
        predictions = self.model.predict(input_image)

        # Assuming you have a list of class labels
        class_labels = [chr(ord('a') + i) for i in range(26)]
        print(class_labels)

        # Get the predicted class
        predicted_class_index = np.argmax(predictions)
        print(predictions)
        print(predicted_class_index)
        predicted_class = class_labels[predicted_class_index]
        print("Predicted Class:", predicted_class)
        return predicted_class
    
    def detect_sign2(self, image_path):


        # Preprocess the image
        input_image = self.preprocess_image2(image_path)
        # Make predictions using the loaded model
        predictions = self.model.predict(input_image)

        # Assuming you have a list of class labels
        class_labels = [chr(ord('a') + i) for i in range(26)]
        print(class_labels)

        # Get the predicted class
        predicted_class_index = np.argmax(predictions)
        print(predictions)
        print(predicted_class_index)
        predicted_class = class_labels[predicted_class_index]
        print("Predicted Class:", predicted_class)
        return predicted_class
    def detect_sign_ann(self,data):
        class_labels = [chr(ord('a') + i) for i in range(26)]
        pred=np.array(self.model(data))
        pred=class_labels[pred.argmax()]
        return pred