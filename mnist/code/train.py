
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

def create_model():  
    model = Sequential([
        Dense(784, activation='sigmoid'),
        Dense(32, activation='sigmoid'),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_test, y_test):
    model = create_model()
    
    history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=True)
    model.summary()
    loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
    
    print(f'Loss: {loss:.3}, Accuracy: {accuracy:.3}')
