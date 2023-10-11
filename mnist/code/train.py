
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot as plt 

model_path = Path('model') / '001'
image_size = 28 * 28
def create(no_features):  
    model = Sequential([
        Dense(32, activation='sigmoid'),
        Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=(None,no_features))
    return model

def train(X_train, y_train, X_test, y_test):
    model = create(X_train.shape[1])
    model.summary()
    
    history = model.fit(X_train, y_train, batch_size=18, epochs=5, validation_split=.1, verbose=True)
    model.save(model_path)

    # loss, accuracy = model.evaluate(X_test, y_test, verbose=True)

    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.plot(accuracy)

    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['training', 'validation'], loc='best')
    # plt.show()

    # print(f'Loss: {loss:.3}, Accuracy: {accuracy:.3}')
