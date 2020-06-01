######## Preparation ########
print('Preparation...')
from tensorflow.keras.datasets import reuters
######## Preparation ######## END

######## Load data ########
print('Load data...')
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print('train_data:', str(len(train_data)))    # 8982
print('test_data:', str(len(test_data)))    # 2246
######## Load data ######## END

######## Data convert ########
print('Data convert to string...')
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join(
                            [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
                            )
print(decoded_newswire)
######## Data convert ######## END
