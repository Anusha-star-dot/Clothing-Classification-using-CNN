__GL_APP_COPY__:from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Step 1: Load and preprocess the dataset
def load_and_preprocess():
    """
    Load image dataset and prepare training, validation, and test data.
    """
    from tensorflow.keras.datasets import fashion_mnist
    from sklearn.model_selection import train_test_split 
    #  Load Fashion MNIST dataset from TensorFlow
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    #  Separate training and test data
    (X_train, y_train), (X_test, y_test)

    #  Normalize image pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    

    # Add channel dimension to images
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    #  Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Return train, validation, and test sets
    return X_train, X_val, X_test, y_train, y_val, y_test


# Step 2: Build the CNN model
def build_cnn(input_shape):
    """
    Define and compile a Convolutional Neural Network.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    #  Initialize a Sequential model
    model = Sequential()

    #  Add first convolution layer
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    
    #  Add pooling layer
    model.add(MaxPooling2D(2, 2))

    #  Add second convolution layer
    model.add(Conv2D(64, (3,3), activation='relu'))

    #  Add second pooling layer
    model.add(MaxPooling2D(2,2))
    #  Flatten feature maps
    model.add(Flatten())
    # Add fully connected (Dense) layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    #  Add output layer
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Step 3: Train the model
def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the CNN using training data and validate on validation data.
    """

    #  Train model using training data
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64
    )
    return model
# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using test data.
    """

    # Evaluate model on test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuarcy: {accuracy}")

    # Generate predictions on test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Print classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_classes))

    #  Print confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    print("\nConfusion Matrix:\n", cm)

    


# Main function
def main():
    # Load and preprocess dataset
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess()

    # Build CNN model
    model = build_cnn((28, 28, 1))

    # Train the CNN model
    model = train_model(model, X_train, y_train, X_val, y_val)
    #  Evaluate the trained model
    evaluate_model(model, X_test, y_test)
    


if __name__ == "__main__":
    main()