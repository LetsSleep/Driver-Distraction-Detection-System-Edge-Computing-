import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model


# model = load_model("models/keras_model.h5")
CLASS_LABELS = ['drinking', 'looking_behind', 'normal', 'on_call']

class TfLiteClassificationModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._inputDetails = self.interpreter.get_input_details()
        self._outputDetails = self.interpreter.get_output_details()

    def predict(self, image):
        """
        args:
          image: a (1, imageHeight, imageWidth, 3) np.array

        Returns array of each class probability
        """

        self.interpreter.set_tensor(self._inputDetails[0]["index"], image)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self._outputDetails[0]["index"])
        
        return output
    
model = TfLiteClassificationModel("model_unquant.tflite")


cap = cv2.VideoCapture(0)

def classify_image(image):
    # Preprocess the image
    image = cv2.resize(image, (100, 100))
    image = image / 255.0

   # Add batch dimension: model expects a 4D input (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)

    # Predict the class of the image
    predictions = model.predict(image, verbose=0)

    # Get the class label with the highest probability
    label = np.argmax(predictions, axis=-1)

    return CLASS_LABELS[label[0]]
    return label[0]

# Capture and classify images in real-time
while True:
    ret, frame = cap.read()

    # Classify the image
    label = classify_image(frame)

    # Display the classified image
    cv2.putText(frame, str(label), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Image Classification', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

