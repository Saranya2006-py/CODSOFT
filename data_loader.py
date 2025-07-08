import string
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

# Load and clean descriptions
def load_descriptions(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    descriptions = {}
    for line in lines:
        if len(line.strip()) < 2 or line.startswith("image"):
            continue  # Skip empty or header lines
        tokens = line.strip().split()
        if len(tokens) < 2:
            continue  # Skip malformed lines
        img_id = tokens[0].split('.')[0]
        desc = ' '.join(tokens[1:])
        desc = desc.translate(str.maketrans('', '', string.punctuation)).lower()
        if img_id not in descriptions:
            descriptions[img_id] = []
        descriptions[img_id].append('startseq ' + desc + ' endseq')
    return descriptions

# Tokenize all descriptions
def create_tokenizer(descriptions):
    all_desc = [d for descs in descriptions.values() for d in descs]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc)
    return tokenizer

# Get max caption length
def max_length(descriptions):
    return max(len(d.split()) for descs in descriptions.values() for d in descs)

# Create input-output sequences (memory-efficient)
def create_sequences(tokenizer, max_len, descriptions, features, vocab_size):
    X1, X2, y = [], [], []
    for key, desc_list in descriptions.items():
        if key not in features:
            continue  # Skip if no feature for this image
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq_index = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                X1.append(features[key])
                X2.append(in_seq)
                y.append(out_seq_index)  # Use integer instead of one-hot
    return np.array(X1), np.array(X2), np.array(y)

# Load everything together
def load_dataset():
    descriptions = load_descriptions('./data/Flicker8k_text/Flickr8k.token.txt')
    with open('./features/image_features.pkl', 'rb') as f:
        features = pickle.load(f)
    tokenizer = create_tokenizer(descriptions)
    max_len = max_length(descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    X1, X2, y = create_sequences(tokenizer, max_len, descriptions, features, vocab_size)
    return X1, X2, y, tokenizer, max_len, vocab_size, descriptions, features
