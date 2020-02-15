import numpy as np
import pandas as pd
import pickle
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import re
import itertools
try:
    with open('data.pickle', 'rb') as file:
        xs, xs_, ys, y_binary, char2int, int2char, chartoint, processed, unique_letters = pickle.load(
            file)
except FileNotFoundException:
    data = pd.read_csv('world-cities.csv')
    cities = data['name']
    processed = []
    for city in cities:
        arr = []
        pattern = re.compile(r'[^a-zA-Z]')
        city = pattern.sub('', city)

        for c in city:
            arr.append(c)

            processed.append(arr)

    processed = [sublist[0]
                 for sublist in itertools.groupby(sorted(processed))]

    unique_letters = []
    for uni in processed:

        for letter in uni:
            unique_letters.append(letter)

    unique_letters = sorted(set(unique_letters))

    char2int = {c: i for i, c in enumerate(unique_letters)}
    int2char = {i: c for i, c in enumerate(unique_letters)}
    training_data = []
    testing_data = []
    seq = 5
    xs = []
    ys = []
    for city in range(len(processed)):

        for i in range(0, len(processed[city]) - seq):
            training_data.append(processed[city][i:i+seq])
            testing_data.append(processed[city][i+seq])

    for city in training_data:
        xs.append([char2int[l] / len(unique_letters) for l in city])

    for city in testing_data:
        ys.append([char2int[l] for l in city])

    xs_ = np.array(xs).reshape((len(xs), seq, 1))
    ys = np.array(ys)
    y_binary = np_utils.to_categorical(ys)
    with open('data.pickle', 'wb') as file:
        pickle.dump((xs, xs_, ys, y_binary, char2int, int2char,
                     chartoint, processed, unique_letters), file)


model = Sequential()
model.add(LSTM(256, input_shape=(
    xs_.shape[1], xs_.shape[2]), return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(len(unique_letters), activation='softmax'))

filepath = "model_weights_saved.hdf5"
try:
    model.load_weights('model_weights_saved.hdf5')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

except FileNotFoundException:
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        desired_callbacks = [checkpoint]
        model.fit(xs_, y_binary, epochs=100, callbacks=desired_callbacks)

def generate(n:int):
        join = []
        for i in range(6):
                index = np.random.randint(0, len(processed)-1)
                a = [char2int[processed[index][a]]for a in range(len(processed[index]))]
                if len(a) < 5:
                        for _ in range(5-len(a)):
                                a.append(np.random.randint(0, len(unique_letters)))
                x = []
                p = []
                x.append(a[:5])
                p.append(x)
                p = np.reshape(p, (1, 5, 1))
                p = p / len(unique_letters)
                pred = model.predict(p)
                i = np.argmax(pred)
                join.append(int2char[i])
        print("".join(join))
generate(np.random.randint(5, 11))
