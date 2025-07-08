import pickle
from tensorflow.keras.models import load_model
from utils.data_loader import load_dataset
from utils.caption_generator import evaluate_model

# --- Step 1: Load model and tokenizer ---
print("[INFO] Loading model and tokenizer...")
model_path = './checkpoints/model_epoch_20.h5'
tokenizer_path = './checkpoints/tokenizer.pkl'

try:
    model = load_model(model_path)
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    exit()

try:
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print(f"[ERROR] Could not load tokenizer: {e}")
    exit()

# --- Step 2: Load dataset ---
print("[INFO] Loading dataset...")
_, _, _, _, max_len, _, descriptions, features = load_dataset()

# Optional sanity check
if not descriptions or not features:
    print("[WARNING] Descriptions or features are empty!")

# --- Step 3: Evaluate the model using BLEU scores ---
print("[INFO] Evaluating model...")
evaluate_model(model, descriptions, features, tokenizer, max_len)
