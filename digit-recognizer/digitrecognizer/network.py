import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

DEFAULT = [[0]] * (28 * 28)

if __name__ == '__main__':
    inputs = layers.Input(shape=(784,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    predictions = layers.Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    estimator = tf.keras.estimator.model_to_estimator(keras_model=model)

    def train_input_fn(shuffle_seed, batch_size, num_epochs, num_parallel_read=4):
        return tf.data.experimental.make_csv_dataset(
            './input/train*.csv',
            batch_size,
            label_name='label',
            num_epochs=num_epochs,
            num_parallel_reads=num_parallel_read,
            shuffle_seed=shuffle_seed
        )

    def eval_input_fn():
        pass

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
