from tensorflow.keras.datasets import mnist
import pandas as pd

# Load the MNIST dataset
# MNIST dataset is pre-splitted: 60,000 for train and 10,000 for test
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Convert the images to a pandas DataFrame; removed label for the train, since the goal of this project is to label properly
df_train = pd.DataFrame(x_train.reshape(60000, -1))  # flatten 28x28 to 784

df_test = pd.DataFrame(x_test.reshape(10000, -1))  # flatten 28x28 to 784
df_test['label'] = pd.DataFrame(y_test)

import matplotlib.pyplot as plt

# Leave this code in case you want to see what MNIST dataset look like
# Not going to run this for fast-running
# plt.imshow(x_train[0], cmap='gray')
# plt.title(f'Label: {y_train[0]}')
# plt.show()

# store them in csv files
output_train_file = "./MNIST_train"
output_test_file = "./MNIST_test"
df_train.to_csv(output_train_file)
df_test.to_csv(output_test_file)