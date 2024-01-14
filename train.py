import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH
from constants import OUTPUT_UNITS, NUM_UNITS, LOSS, LEARNING_RATE, EPOCHS, BATCH_SIZE, SAVE_MODEL_PATH


def build_model(output_units, num_units, loss, learning_rate):
    # Create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)
    model = keras.Model(input, output)

    # Compile model
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )
    model.summary()
    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    # Generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # Build the network
    model = build_model(output_units, num_units, loss, learning_rate)

    # Train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()
