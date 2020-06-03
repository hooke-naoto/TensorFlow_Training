######## Preparation ########
print('Preparation...')
from tensorflow.keras.datasets import boston_housing
import numpy as np
import tensorflow as tf
import time
######## Preparation ######## END

######## Load data ########
print('Load data...')
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print('train_data:', str(train_data.shape))    # (404, 13)
print('test_data:', str(test_data.shape))    # (102, 13)
print('train_targets:', str(train_targets.shape))    # (404,)
print('test_targets:', str(test_targets.shape))    # (102,)
######## Load data ######## END

######## Normalization ########
print('Normilization...')
train_data_mean = train_data.mean(axis=0)
train_data_std = train_data.std(axis=0)
train_data -= train_data_mean
train_data /= train_data_std
test_data  -= train_data_mean
test_data  /= train_data_std
######## Normalization ######## END

######## NN definiion ########
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)),    # train_data.shape[1] = 13
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

######## k-fold cross-validation ########
# print('k-fold cross-validation...')
# k = 4
# num_validation_samples = len(train_data) // k    # Division as the round down.
# num_epochs = 500
# mae_history_all = []
# time_all = []
# for i in range(k):
#     print('Fold:', str(i + 1), '/', str(k))
#     time_start = time.time()
#     train_data_validation = train_data[i*num_validation_samples:(i + 1)*num_validation_samples]
#     train_targets_validation = train_targets[i*num_validation_samples:(i + 1)*num_validation_samples]
#     train_data_partial = np.concatenate([train_data[ :i*num_validation_samples], train_data[(i + 1)*num_validation_samples: ]], axis=0)
#     train_targets_partial = np.concatenate([train_targets[ :i*num_validation_samples], train_targets[(i + 1)*num_validation_samples: ]], axis=0)
#     print('NN definition...')
#     model = build_model()
#     print('NN training...')
#     history = model.fit(
#         train_data_partial,
#         train_targets_partial,
#         validation_data=(train_data_validation, train_targets_validation),
#         epochs=num_epochs,
#         batch_size=1,
#         verbose=0
#         )
#     # print(history.history.keys())
#     mae_history = history.history['val_mae']
#     mae_history_all.append(mae_history)
#     time_all.append(time.time() - time_start)
# mae_history_mean = [ np.mean( [ x[i] for x in mae_history_all ] ) for i in range(num_epochs) ]    # MAE average values as each epoch.
# print('Time:', str(time_all))    # [Time1, Time2, Time3, Time4]
# print('MAE mean number:', str(len(mae_history_mean)))    # 500
# # print('MAE mean data:', str(mae_history_mean))
# # print('MAE all data:', str(mae_history_all))    # 4 x 500
# # Show graph
# import matplotlib.pyplot as plt
# def smooth_curve(points, factor=0.9):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous*factor + point*(1 - factor))
#         else:
#             smoothed_points.append(point)
#     return smoothed_points
# mae_history_smooth = smooth_curve(mae_history_mean[10: ])    # Remove first 10 points becuase not significant.
# plt.plot(range(1, len(mae_history_smooth) + 1), mae_history_smooth)
# plt.xlabel('Epochs')
# plt.xlabel('Validation MAE')
# plt.show()
######## k-fold cross validation ######## END

######## NN training final ########
print('NN training final...')
time1 = time.time()
model = build_model()
time2 = time.time()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
time3 = time.time()
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
time4 = time.time()
print('Time for build_model:', str(time2 - time1))
print('Time for model.fit:', str(time3 - time2))
print('Time for model.evaluate:', str(time4 - time3))
print('test_mse_score:', str(test_mse_score))
print('test_mae_score:', str(test_mae_score))
######## NN training final ######## END
