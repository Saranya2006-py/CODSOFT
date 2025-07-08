import os
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
from tqdm import tqdm

dataset_path = './data/Flicker8k_Dataset/'
model = ResNet50(weights='imagenet')
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def extract_features(directory):
    features = {}
    for img_name in tqdm(os.listdir(directory)):
        filename = os.path.join(directory, img_name)
        img = image.load_img(filename, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = model.predict(img, verbose=0)
        img_id = img_name.split('.')[0]
        features[img_id] = feature[0]
    return features

features = extract_features(dataset_path)
with open('./features/image_features.pkl', 'wb') as f:
    pickle.dump(features, f)
