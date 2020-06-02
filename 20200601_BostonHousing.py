######## Preparation ########
print('Preparation...')
from tensorflow.keras.datasets import boston_housing
import numpy as np
import tensorflow as tf
######## Preparation ######## END

######## Load data ########
print('Load data...')
(train_data, train_targets), (test_data, test_labels) = boston_housing.load_data()
print('train_data:', str(train_data.shape))    # 404 x 13
print('test_data:', str(test_data.shape))    # 102 x 13
print('train_targets:', str(train_targets.shape))    # 404 x 1
######## Load data ######## END

######## Normalization ########
print('Normilization...')
train_data_mean = train_data.mean(axis=0)
train_data_std = train_data.std(axis=0)
train_data -= train_data_mean
train_data /= train_data_std
test_data -= train_data_mean
test_data -= train_data_std
######## Normalization ######## END

######## NN definiion ########
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)    # No Activation! This output for regression.
        ])
    # Model compile
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
        loss=tf.keras.losses.mse,
        metrics=['mae'])
    return model
######## NN definiion ######## END

######## Preparation for k-fold cross-validation ########
print('k-fold cross-validation...')
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
mae_history_all = []
for i in range(k):
    print('Fold:', i)
    val_data = train_data[i*num_val_samples:(i + 1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i + 1)*num_val_samples]
    train_data_partial = np.concatenate([train_data[ :i*num_val_samples], train_data[(i + 1)*num_val_samples: ]], axis=0)
    train_targets_partial = np.concatenate([train_targets[ :i*num_val_samples], train_targets[(i + 1)*num_val_samples: ]], axis=0)
    print('NN definition...')
    model = build_model()
    print('NN training...')
    history = model.fit(train_data_partial, train_targets_partial, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1, verbose=0)
    # print(history.history.keys())
    mae_history = history.history['val_mae']
    mae_history_all.append(mae_history)
mae_history_mean = [np.mean([x[i] for x in mae_history_all]) for i in range(num_epochs)]
print('Mean of all scores:', str(mae_history_mean))
print('All socres:', str(mae_history_all))
import matplotlib.pyplot as plt
plt.plot(range(1, len(mae_history_mean) + 1), mae_history_mean)
plt.xlabel('Epochs')
plt.xlabel('Validation MAE')
plt.show()
######## Preparation for k-fold cross validation ######## END
