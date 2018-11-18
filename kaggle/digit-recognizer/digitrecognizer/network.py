import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


if __name__ == '__main__':
    images = pd.read_csv('./input/train.csv')
    labels = images[['label']]
    images = images.drop(columns=['label']) / 255.0

    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    estimator = tf.keras.estimator.model_to_estimator(keras_model=model)

    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=images,
        y=labels,
        batch_size=128,
        num_epochs=1,
        shuffle=True,
    )

estimator.train(input_fn=train_input_fn, steps=10)
