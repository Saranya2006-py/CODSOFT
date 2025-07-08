from models.caption_model import define_model
from utils.data_loader import load_dataset
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Load data
print("[INFO] Loading dataset...")
X1, X2, y, tokenizer, max_len, vocab_size, _, _ = load_dataset()

# Check for checkpoints
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Look for latest checkpoint
existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.h5')]
if existing_checkpoints:
    latest_checkpoint = sorted(existing_checkpoints)[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path)
    initial_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
else:
    print("[INFO] Starting from scratch...")
    model = define_model(vocab_size, max_len)
    initial_epoch = 0

# Compile model (use sparse loss because y is integer encoded)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Setup checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.h5'),
    save_freq='epoch',
    save_best_only=False,
    verbose=1
)

# Train
print("[INFO] Training model...")
model.fit(
    [X1, X2], y,
    epochs=20,
    initial_epoch=initial_epoch,
    batch_size=64,
    callbacks=[checkpoint_callback],
    verbose=1
)

# Save tokenizer separately
with open(os.path.join(checkpoint_dir, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)

print("[INFO] Training complete. Model and tokenizer saved.")
