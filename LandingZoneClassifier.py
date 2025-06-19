import unrealcv

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class Classifier:
    def __init__(self, model_path, image_size=(224, 224)):
        self.model = load_model(model_path)
        self.image_size = image_size
        self.client = unrealcv.Client(('localhost', 5000))
        self._connect_unrealcv()

    def _connect_unrealcv(self):
        self.client.connect()
        if not self.client.isconnected():
            print('UnrealCV connection failed')
        else:
            print('Connected to UnrealCV')

    def take_screenshot(self, image_path='screenshot.png'):
         print("[DEBUG] Inside take_screenshot")
        
        
        
        
         if not self.client.isconnected():
            print('UnrealCV is not connected')
            return None
         print("[DEBUG] Sending screenshot request to UE")
         self.client.request('vget /camera/0/lit ' + image_path)
         print(f"[DEBUG] Screenshot saved to: {image_path}")
         return image_path

    def preprocess_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.image_size)
        img = preprocess_input(img)
        img = tf.expand_dims(img, axis=0)
        return img

    def predict_single_image(self, image_path):
        preprocessed_img = self.preprocess_image(image_path)
        prediction = self.model.predict(preprocessed_img)
        probability = prediction[0][0]
        predicted_class = "safe" if probability > 0.5 else "unsafe"
        return predicted_class, probability

    def should_run_episode(self,model):
        screenshot_path = self.take_screenshot()
        if screenshot_path is None:
            return False
        label, probability = self.predict_single_image(screenshot_path)
        print(f"Landing zone classified as: {label} (Confidence: {probability:.2f})")
        return label == 'safe'

