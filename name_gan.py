"""GAN implementation for name generation."""
import time
import json
import os
import tensorflow as tf

from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Reshape,
    Conv1D,
    Conv1DTranspose,
    Flatten,
    Dropout,
)
from tensorflow.keras.models import Sequential
import numpy as np


from IPython import display


EPOCHS = 5000
MULTIPLIER = 5
CHECKPOINT_DIR = "./training_checkpoints"


def create_letter_list(names):
    return list(sorted(set("".join(set(names)))))


def make_generator_model():
    model = Sequential()
    model.add(Dense(input_dim, use_bias=False, input_shape=(input_dim,),))
    ##    print(model.output_shape)
    # model.add(BatchNormalization())
    # Dense(max_length * 2, use_bias=False)
    ##    print(model.output_shape)
    # model.add(BatchNormalization())
    # Dense(max_length, activation="tanh", use_bias=False)
    # print(model.output_shape)
    # model.add(BatchNormalization())

    model.add(Reshape((1, input_dim)))
    print(model.output_shape)

    model.add(
        Conv1DTranspose(
            input_dim,
            (4),
            activation="elu",
            strides=(1),
            padding="same",
            use_bias=False,
        )
    )
    # print(model.output_shape)
    model.add(BatchNormalization())
    model.add(
        Conv1DTranspose(
            max_length * 4,
            (4),
            activation="elu",
            strides=(1),
            padding="same",
            use_bias=False,
        )
    )
    # print(model.output_shape)
    model.add(BatchNormalization(momentum=0.8))
    model.add(
        Conv1DTranspose(
            max_length * 2,
            (4),
            strides=(1),
            activation="elu",
            padding="same",
            use_bias=False,
        )
    )
    # print(model.output_shape)
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(max_length, activation="tanh", use_bias=False))
    model.add(Reshape((max_length,)))
    print("output_shape", model.output_shape)
    assert model.output_shape == (None, max_length,)  # Note: None is the batch size

    return model


def make_discriminator_model():
    model = Sequential()
    # model.add(Dense(max_length * 8, use_bias=False, input_shape=(input_dim, 1),))
    #    print(model.output_shape)
    # Dense(max_length * 4, use_bias=False)
    #    print(model.output_shape)
    # Dense(max_length * 2, activation="elu", use_bias=False)
    model.add(
        Conv1D(
            30 * max_length,
            (4),
            kernel_initializer="he_uniform",
            strides=(1),
            activation="elu",
            padding="same",
            input_shape=[max_length, 1],
        )
    )
    # print(model.output_shape)
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(
        Conv1D(
            5 * max_length,
            (4),
            kernel_initializer="he_uniform",
            activation="elu",
            strides=(1),
            padding="same",
        )
    )
    # print(model.output_shape)
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))
    print(model.output_shape)

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for name_batch in dataset:
            name_batch = tf.expand_dims(name_batch, -1)
            #            print(name_batch)
            train_step(name_batch)
        seed = tf.random.normal([1, input_dim])
        generate_and_save_names(generator, seed)
        #        print(seed)

        # Save the model every x epochs
        if (epoch + 1) % int(EPOCHS / 4) == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

    generate_and_save_names(generator, seed)
    display.clear_output(wait=True)


def generate_and_save_names(model, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    with open("data/predictions.txt", mode="a+") as pred_file:
        for pred in predictions:
            new_name_int = np.array(pred)
            new_name_int = (new_name_int * (len(letters) / 2)) + (len(letters) / 2)
            decision = discriminator(tf.expand_dims(predictions, -1))
            print("decision", decision.numpy()[0][0])
            try:
                new_name = "".join(
                    [letters[int(np.round(number)) - 1] for number in new_name_int]
                )
                pred_file.write(f"{new_name, decision.numpy()[0][0]}\n")
            except IndexError:
                pred_file.write(f"Out of bounds... {decision.numpy()[0][0]}\n")


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

names = []
with open("data/names.json", "r") as name_file:
    name_dict = json.load(name_file)
names = list(name_dict.keys())[9000:10000]

max_length = max(len(name) for name in names)
print(max_length)
string_shape = (max_length, 1)
standard_names = [name + "_" * (max_length - len(name)) for name in names]
letters = create_letter_list(standard_names)
print(letters)
name_ints = []
for standard_name in standard_names:
    name_int = [letters.index(character) + 1 for character in standard_name]
    name_int = list((np.array(name_int) - len(letters) / 2) / (len(letters) / 2))
    name_ints.append(name_int)
letter_ints = list(set(letter_int for name in name_ints for letter_int in name))
print(min(letter_ints), max(letter_ints))
batch_size = int(len(name_ints) / 100)

buffer_size = len(name_ints)
input_dim = max_length * MULTIPLIER

train_dataset = (
    tf.data.Dataset.from_tensor_slices(name_ints).shuffle(buffer_size).batch(batch_size)
)

generator = make_generator_model()
noise = tf.random.normal([1, input_dim])
print("noise", noise.shape)
generated_name = generator(noise, training=False)
print("name", generated_name.shape)

discriminator = make_discriminator_model()
decision = discriminator(tf.expand_dims(generated_name, -1))
print("disc", tf.expand_dims(generated_name, -1).shape)
print("decision", decision.numpy()[0][0])


checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)
checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))

num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([1, input_dim])
# seed = tf.random.normal([1, 15], int(len(letters)/2), 1)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(names):
    noise = tf.random.normal([batch_size, input_dim])
    #    print("noise", noise.shape)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_names = tf.expand_dims(generator(noise, training=True), -1)
        #        print("gen:", generated_names.shape)
        #        print("names", names.shape)
        real_output = discriminator(names, training=True)
        fake_output = discriminator(generated_names, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )


train(train_dataset, EPOCHS)
