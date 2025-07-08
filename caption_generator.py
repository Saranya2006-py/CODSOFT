# utils/caption_generator.py

import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

from utils.generator_core import generate_caption  # ✅ FIXED IMPORT

def extract_feature(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img).reshape(1, 2048)

# Load caption model and tokenizer
print("[INFO] Loading model and tokenizer...")
model = load_model('./checkpoints/model_epoch_20.h5')
  # Update if needed
with open('./checkpoints/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
max_len = 37  # You may load this dynamically if stored elsewhere

# Load ResNet50 for feature extraction
print("[INFO] Loading ResNet50 feature extractor...")
resnet = ResNet50(weights='imagenet')
resnet = Model(inputs=resnet.inputs, outputs=resnet.layers[-2].output)

# Image path to test
img_path = './data/Flicker8k_Dataset/54723805_bcf7af3f16.jpg'
print(f"[INFO] Extracting features from image: {img_path}")
feature = extract_feature(img_path, resnet)

# Generate caption
print("[INFO] Generating caption...")
caption = generate_caption(model, tokenizer, feature, max_len)
print('Caption:', caption)

# Show image with caption
img = image.load_img(img_path)
plt.imshow(img)
plt.title(caption)
plt.axis('off')
plt.show()
