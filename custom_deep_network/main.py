import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


class Cifar10Classification:
    def __init__(self):
        self.load_data()

    def load_data(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()
        # Normalize pixel values to be between 0 and 1
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0
    
    def make_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))

    def model_summery(self):
        print(self.model.summary())
    
    def model_run(self):
        self.make_model()
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        self.history = self.model.fit(self.train_images, self.train_labels, epochs=10, validation_data=(self.test_images, self.test_labels))  
        self.model_test()
        self.visualize()

    def model_test(self):
        self.test_loss, self.test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)
        print("test loss: ",self.test_loss)
        print("test accurayc: ",self.test_acc)

    def visualize(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()



obj=Cifar10Classification()
obj.model_run()
