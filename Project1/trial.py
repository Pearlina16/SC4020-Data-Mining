# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the dataset
def preprocess_images(images):
    images = images.astype('float32') / 255.0
    images = np.expand_dims(images, axis=-1)  # Add channel dimension (28, 28) -> (28, 28, 1)
    return images

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

# Create pairs of images and labels for the Siamese network
def create_pairs(images, labels):
    pairs = []
    labels_pair = []
    
    num_classes = max(labels) + 1
    digit_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    for idx in range(len(images)):
        current_image = images[idx]
        label = labels[idx]
        
        # Choose a positive pair (same class)
        pos_idx = np.random.choice(digit_indices[label])
        pos_image = images[pos_idx]
        
        # Choose a negative pair (different class)
        neg_label = np.random.choice([i for i in range(num_classes) if i != label])
        neg_idx = np.random.choice(digit_indices[neg_label])
        neg_image = images[neg_idx]
        
        # Add positive and negative pairs
        pairs += [[current_image, pos_image], [current_image, neg_image]]
        labels_pair += [1, 0]  # 1 for similar, 0 for dissimilar
    
    return np.array(pairs), np.array(labels_pair)

train_pairs, train_pair_labels = create_pairs(train_images, train_labels)
test_pairs, test_pair_labels = create_pairs(test_images, test_labels)

# Define the Siamese network model architecture
def create_siamese_network(input_shape):
    # Input layers
    input = layers.Input(shape=input_shape)
    
    # Convolutional base (shared by both inputs)
    convnet = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu')
    ])
    
    # Use the same convolutional base for both inputs
    encoded_left = convnet(input)
    encoded_right = convnet(input)
    
    # Combine the outputs of the two inputs using a Lambda layer with L1 distance
    l1_distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([encoded_left, encoded_right])
    
    # Output layer with a single neuron (similarity score)
    output = layers.Dense(1, activation='sigmoid')(l1_distance)
    
    # Define the model
    siamese_model = Model([input, input], output)
    
    return siamese_model

# Compile and train the Siamese network
input_shape = (28, 28, 1)
siamese_model = create_siamese_network(input_shape)
siamese_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Reshape pairs for training
train_pair_left = train_pairs[:, 0].reshape(-1, 28, 28, 1)
train_pair_right = train_pairs[:, 1].reshape(-1, 28, 28, 1)

test_pair_left = test_pairs[:, 0].reshape(-1, 28, 28, 1)
test_pair_right = test_pairs[:, 1].reshape(-1, 28, 28, 1)

# Train the model
history = siamese_model.fit(
    [train_pair_left, train_pair_right], 
    train_pair_labels, 
    validation_data=([test_pair_left, test_pair_right], test_pair_labels),
    batch_size=64, 
    epochs=10
)

# Plot accuracy and loss curves
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test model on a new pair
def test_siamese(model, pair):
    pair_left = pair[0].reshape(1, 28, 28, 1)
    pair_right = pair[1].reshape(1, 28, 28, 1)
    return model.predict([pair_left, pair_right])

# Test a pair of images
sample_pair = test_pairs[0]
prediction = test_siamese(siamese_model, sample_pair)
print(f'Similarity score: {prediction[0][0]}')
