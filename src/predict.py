import cv2
import numpy as np
from tensorflow.keras.models import load_model

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DELETE", "NOTHING"]

def predict_image(image_path):
    model = load_model("saved_model/asl_model.h5")

    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.reshape(img, (1, 64, 64, 3))

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    return CLASSES[class_index]
