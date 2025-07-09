from keras.datasets import mnist
import pandas as pd
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert the first few images and labels to a pandas DataFrame
df_train = pd.DataFrame(x_train[:5].reshape(5, -1))  # flatten 28x28 to 784
df_train['label'] = y_train[:5]

plt.imshow(x_train[0], cmap='gray')
plt.title(f'Label: {y_train[0]}')
plt.show()