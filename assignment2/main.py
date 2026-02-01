import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. DATA PREPROCESSING
def load_and_prep_data(model_type='CNN'):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize pixel values to be between 0 and 1 [cite: 8]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    if model_type == 'CNN':
        # Reshape for CNN: (28, 28, 1) [cite: 11]
        x_train = x_train.reshape((-1, 28, 28, 1))
        x_test = x_test.reshape((-1, 28, 28, 1))
    else:
        # Reshape for MLP: (784,) [cite: 16]
        x_train = x_train.reshape((-1, 784))
        x_test = x_test.reshape((-1, 784))
    return x_train, y_train, x_test, y_test

# 2. MODEL BUILDERS
def build_cnn(activation='relu', optimizer='adam', dropout_rate=0.25, use_bn=False):
    model = models.Sequential()
    # Conv Layers [cite: 12]
    model.add(layers.Conv2D(32, (3, 3), activation=activation, input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation=activation))
    model.add(layers.MaxPooling2D((2, 2)))
    
    if use_bn: model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate)) # [cite: 13]
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation)) # [cite: 14]
    model.add(layers.Dense(10, activation='softmax')) # [cite: 14]
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_mlp(hidden_units=[256, 128], optimizer='sgd'):
    model = models.Sequential()
    model.add(layers.Input(shape=(784,))) # [cite: 16]
    for units in hidden_units:
        model.add(layers.Dense(units)) # [cite: 17, 20]
        model.add(layers.BatchNormalization()) # [cite: 18, 21]
        model.add(layers.ReLU()) # [cite: 19, 22]
    model.add(layers.Dense(10, activation='softmax')) # [cite: 24]
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. TASK EXECUTION & LOGGING
results = []

def run_experiment(name, model, x_train, y_train, x_test, y_test, epochs=10):
    print(f"\nRunning Experiment: {name}")
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=0)
    test_acc = history.history['val_accuracy'][-1]
    results.append({'Experiment': name, 'Final Accuracy': test_acc})
    return history

# --- MAIN EXECUTION ---
x_train_cnn, y_train, x_test_cnn, y_test = load_and_prep_data('CNN')
x_train_mlp, _, x_test_mlp, _ = load_and_prep_data('MLP')

# Task 1: Activation Challenge (CNN) [cite: 27]
h_sigmoid = run_experiment("CNN-Sigmoid", build_cnn(activation='sigmoid', optimizer='sgd'), x_train_cnn, y_train, x_test_cnn, y_test)
h_tanh = run_experiment("CNN-Tanh", build_cnn(activation='tanh', optimizer='sgd'), x_train_cnn, y_train, x_test_cnn, y_test)
h_relu = run_experiment("CNN-ReLU", build_cnn(activation='relu', optimizer='sgd'), x_train_cnn, y_train, x_test_cnn, y_test)

# Task 2: Optimizer Showdown (Best activation: ReLU) [cite: 31]
run_experiment("CNN-Adam", build_cnn(optimizer='adam'), x_train_cnn, y_train, x_test_cnn, y_test)
run_experiment("CNN-SGD-Momentum", build_cnn(optimizer=optimizers.SGD(momentum=0.9)), x_train_cnn, y_train, x_test_cnn, y_test)

# Task 3: Regularization Contrast [cite: 36, 37]
run_experiment("No BN/Dropout", build_cnn(dropout_rate=0.0, use_bn=False), x_train_cnn, y_train, x_test_cnn, y_test)
run_experiment("Dropout 0.1, No BN", build_cnn(dropout_rate=0.1, use_bn=False), x_train_cnn, y_train, x_test_cnn, y_test)
run_experiment("Dropout 0.25 + BN", build_cnn(dropout_rate=0.25, use_bn=True), x_train_cnn, y_train, x_test_cnn, y_test)

# Required Table 1 Comparisons [cite: 26]
run_experiment("CNN-1 (Adam, 10 ep)", build_cnn(optimizer='adam'), x_train_cnn, y_train, x_test_cnn, y_test, epochs=10)
run_experiment("MLP-1 (SGD, 20 ep)", build_mlp(hidden_units=[512, 256, 128], optimizer='sgd'), x_train_mlp, y_train, x_test_mlp, y_test, epochs=20)
run_experiment("MLP-2 (Adam, 15 ep)", build_mlp(hidden_units=[256], optimizer='adam'), x_train_mlp, y_train, x_test_mlp, y_test, epochs=15)

# 4. OUTPUTS (Comparison Table) [cite: 41]
df_results = pd.DataFrame(results)
print("\n--- FINAL COMPARISON TABLE ---")
print(df_results)

# 5. VISUALIZATION (Task 1 Example) [cite: 42]
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(h_sigmoid.history['val_accuracy'], label='Sigmoid')
plt.plot(h_tanh.history['val_accuracy'], label='Tanh')
plt.plot(h_relu.history['val_accuracy'], label='ReLU')
plt.title('Validation Accuracy: Activation Functions')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(h_sigmoid.history['val_loss'], label='Sigmoid')
plt.plot(h_tanh.history['val_loss'], label='Tanh')
plt.plot(h_relu.history['val_loss'], label='ReLU')
plt.title('Validation Loss: Activation Functions')
plt.xlabel('Epochs')
plt.legend()
plt.show()