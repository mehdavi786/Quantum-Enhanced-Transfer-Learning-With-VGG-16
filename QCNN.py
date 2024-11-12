pip install tensorflow tensorflow-quantum pennylane
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Define a quantum layer function
def quantum_layer(qubits, circuit):
    # Create a quantum layer by applying a parameterized quantum circuit (PQC)
    return tfq.layers.PQC(circuit, qubits)

# Quantum Feature Map Circuit: example of a simple quantum feature map
def quantum_feature_map(qubits, features):
    circuit = cirq.Circuit()
    
    # Apply a quantum feature map based on input features
    for i, feature in enumerate(features):
        circuit.append(cirq.ry(feature)(qubits[i]))
    
    return circuit

# Build the VGG16 model with quantum layers
def build_vgg16_with_quantum_layers(input_shape=(224, 224, 3), num_classes=10):
    model = models.Sequential()

    # Classical Convolutional Layers
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Flatten to prepare for quantum layer
    model.add(Flatten())

    # Adding a quantum layer
    qubits = cirq.LineQubit.range(4)  # Example qubits (4 qubits for simplicity)
    
    # Create a quantum feature map for each input (features should be mapped from classical data)
    feature_map = quantum_feature_map(qubits, np.random.random(4))  # Example random features

    # Add the quantum layer
    model.add(quantum_layer(qubits, feature_map))

    # Fully connected layers replaced with quantum layers
    model.add(Dropout(0.5))
    model.add(Flatten())  # Flatten for further classical processing

    # Classical Dense Layer for final classification
    model.add(layers.Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Softmax for multi-class classification

    # Compile the model
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Parameters
input_shape = (224, 224, 3)
num_classes = 10  # Change based on your dataset's classes


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define a data generator with augmentation
def create_data_generators(train_dir, val_dir, batch_size=32):
    # Image augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # For validation data, only rescale
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Flow training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Flow validation data
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator

# Define paths for your training and validation data directories
train_dir = 'path_to_train_data'
val_dir = 'path_to_val_data'

# Set batch size and epochs
batch_size = 32
epochs = 10

# Create the data generators
train_generator, validation_generator = create_data_generators(train_dir, val_dir, batch_size)

# Define callbacks for saving the best model and early stopping
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint, early_stopping]
)

# After training, load the best model and evaluate it on the validation set
model.load_weights('best_model.h5')
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation loss: {val_loss}")
print(f"Validation accuracy: {val_accuracy}")

