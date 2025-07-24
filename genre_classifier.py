import os
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Path to the GTZAN genres_origin directory
GENRES_DIR = "genres_origin"
SAMPLE_RATE = 22050
CHUNK_DURATION = 4  # seconds
CHUNK_OVERLAP = 2   # seconds

def get_all_audio_files(genres_dir):
    audio_files = []
    genres = []
    for genre in os.listdir(genres_dir):
        genre_dir = os.path.join(genres_dir, genre)
        if os.path.isdir(genre_dir):
            for file in os.listdir(genre_dir):
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(genre_dir, file))
                    genres.append(genre)
    return audio_files, genres

# 1. Pick a random audio file and visualize it
audio_files, genres = get_all_audio_files(GENRES_DIR)
random_idx = random.randint(0, len(audio_files)-1)
random_audio_path = audio_files[random_idx]
random_genre = genres[random_idx]

y, sr = librosa.load(random_audio_path, sr=SAMPLE_RATE)
duration = librosa.get_duration(y=y, sr=sr)

plt.figure(figsize=(14, 4))
librosa.display.waveshow(y, sr=sr)
plt.title(f"Waveform of {os.path.basename(random_audio_path)} ({random_genre})")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# 2. Chunk the audio into 4s with 2s overlap, visualize a chunk
def chunk_audio(y, sr, chunk_duration, overlap):
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    step = chunk_samples - overlap_samples
    chunks = []
    for start in range(0, len(y) - chunk_samples + 1, step):
        end = start + chunk_samples
        chunks.append(y[start:end])
    return chunks

chunks = chunk_audio(y, sr, CHUNK_DURATION, CHUNK_OVERLAP)
print(f"Total chunks from audio: {len(chunks)}")

# Visualize the first chunk
plt.figure(figsize=(10, 3))
librosa.display.waveshow(chunks[0], sr=sr)
plt.title("Waveform of First 4s Chunk")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# 3. Visualize melspectrogram for entire audio
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram (Full Audio)")
plt.tight_layout()
plt.show()

# 4. Visualize melspectrogram for a chunk
chunk_S = librosa.feature.melspectrogram(y=chunks[0], sr=sr, n_mels=128)
chunk_S_dB = librosa.power_to_db(chunk_S, ref=np.max)
plt.figure(figsize=(8, 3))
librosa.display.specshow(chunk_S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram (First Chunk)")
plt.tight_layout()
plt.show()

# 5. Data Preprocessing: Extract melspectrograms for all chunks of all files
def extract_chunks_and_labels(genres_dir, chunk_duration, overlap, sr):
    X = []
    y = []
    label_map = {}
    label_counter = 0
    for genre in sorted(os.listdir(genres_dir)):
        genre_dir = os.path.join(genres_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
        if genre not in label_map:
            label_map[genre] = label_counter
            label_counter += 1
        for file in os.listdir(genre_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(genre_dir, file)
                audio, _ = librosa.load(file_path, sr=sr)
                chunks = chunk_audio(audio, sr, chunk_duration, overlap)
                for chunk in chunks:
                    S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    # Resize to fixed shape (128, 173) for 4s at 22050Hz
                    if S_dB.shape[1] < 173:
                        pad_width = 173 - S_dB.shape[1]
                        S_dB = np.pad(S_dB, ((0,0),(0,pad_width)), mode='constant')
                    elif S_dB.shape[1] > 173:
                        S_dB = S_dB[:, :173]
                    X.append(S_dB)
                    y.append(label_map[genre])
    return np.array(X), np.array(y), label_map

print("Extracting features from all audio files (this may take a while)...")
X, y, label_map = extract_chunks_and_labels(GENRES_DIR, CHUNK_DURATION, CHUNK_OVERLAP, SAMPLE_RATE)
print(f"Feature shape: {X.shape}, Labels shape: {y.shape}")

# 6. Split data into train and test sets
X = X[..., np.newaxis]  # Add channel dimension
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 7. Build the model
num_classes = len(label_map)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 173, 1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 8. Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 9. Save the model (for Windows users)
model.save("music_genre_classifier.h5")
print("Model saved as music_genre_classifier.h5")

# 10. Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# 11. Visualize accuracy and loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# 12. Precision, Recall, Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=sorted(label_map, key=label_map.get)))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, sorted(label_map, key=label_map.get), rotation=45)
plt.yticks(tick_marks, sorted(label_map, key=label_map.get))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# 13. Test on a new audio file (test music genre)
def predict_genre(audio_path, model, label_map, chunk_duration=4, overlap=2, sr=22050):
    audio, _ = librosa.load(audio_path, sr=sr)
    chunks = chunk_audio(audio, sr, chunk_duration, overlap)
    preds = []
    for chunk in chunks:
        S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        if S_dB.shape[1] < 173:
            pad_width = 173 - S_dB.shape[1]
            S_dB = np.pad(S_dB, ((0,0),(0,pad_width)), mode='constant')
        elif S_dB.shape[1] > 173:
            S_dB = S_dB[:, :173]
        S_dB = S_dB[np.newaxis, ..., np.newaxis]
        pred = model.predict(S_dB)
        preds.append(np.argmax(pred))
    # Majority vote
    inv_label_map = {v: k for k, v in label_map.items()}
    pred_label = max(set(preds), key=preds.count)
    print(f"Predicted genre: {inv_label_map[pred_label]}")
    return inv_label_map[pred_label]

# Example usage for a test file (replace with your test file path)
# test_audio_path = "genres_origin/blues/blues.00000.wav"
# predict_genre(test_audio_path, model, label_map)

print("Music genre classification pipeline complete.")
