from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import os

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
            # Ensure all keypoints have the same shape
            padded_res = np.zeros((21 * 3))  # Assuming each keypoint has 21 coordinates
            if isinstance(res, np.ndarray):
                if res.ndim > 0:
                    padded_res[:min(res.shape[0], 21 * 3)] = res[:min(res.shape[0], 21 * 3)]
            window.append(padded_res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions)).astype(int)  # Ensure to_categorical uses correct number of classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Model architecture
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(50, 63)))  # Assuming 63 coordinates for each keypoint
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(26, activation='softmax'))  # Output layer with 26 units for 26 classes

# Compilation
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7)

# Training
history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[tb_callback, early_stopping, reduce_lr])

# Model summary
model.summary()

# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
