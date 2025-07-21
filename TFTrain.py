from keras.models import Sequential
from keras.layers import (InputLayer, Conv2D, MaxPooling2D, Dropout, BatchNormalization,
                          GlobalAveragePooling2D, Dense, RandomFlip, RandomRotation,
                          RandomZoom, RandomContrast, RandomBrightness, Rescaling)
from keras.optimizers import Adam
from keras.regularizers import L1L2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import image_dataset_from_directory

train_gen = image_dataset_from_directory(
    "Data/train",
    labels="inferred",
    label_mode="categorical",
    image_size=(150, 150),
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_gen = image_dataset_from_directory(
    "Data/val",
    labels="inferred",
    label_mode="categorical",
    image_size=(150, 150),
    batch_size=32,
    shuffle=False
)

print("Train/Validation indices: ", train_gen.class_names)

model = Sequential([
    InputLayer(shape=(150, 150, 3)),
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomContrast(0.2),
    RandomBrightness(0.2),
    Rescaling(1./255),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Dropout(0.3),
    GlobalAveragePooling2D(),
    Dense(512, activation='relu', kernel_regularizer=L1L2(l1=0.0001, l2=0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(train_gen.class_names), activation='softmax')
])

opt = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
best_model = ModelCheckpoint('bestmodel.keras', monitor='val_loss', save_best_only=True)

history = model.fit(train_gen, validation_data=valid_gen, epochs=50, callbacks=[best_model, early_stopping])
