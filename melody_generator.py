import json
import numpy as np
import music21 as m21
import tensorflow.keras as keras
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
from constants import TEMPERATURE, SAVE_MODEL_PATH


class MelodyGenerator:
    def __init__(self, model_path=SAVE_MODEL_PATH):
        try:
            self.model_path = model_path
            self.model = keras.models.load_model(model_path)
        except Exception as e:
            print(f'{SAVE_MODEL_PATH} not found\nException raised: {e}')

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        # Create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # Map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # Limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # One-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))

            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # Max prediction
            probabilities = self.model.predict(onehot_seed)[0]

            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # Update seed
            seed.append(output_int)

            # Map int to our encoding
            output_symbol = [key for key, value in self._mappings.items() if value == output_int][0]

            # Check the end of a melody
            if output_symbol == "/":
                break

            # Update the melody
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilities, temperature):
        # temperature -> infinity
        # temperature -> 0
        # temperature = 1
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))  # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="melody.midi"):
        # Create a m21 stream
        stream = m21.stream.Stream()

        # Parse all the symbols in the melody and create note/rest objects
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            # Handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):
                # Ensure we're dealing with note/rest beyond the first note/rest
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter  # 0.25 * 4 = 1
                    # Handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    # Handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                    stream.append(m21_event)

                    # Reset the step counter
                    step_counter = 1

                start_symbol = symbol
            # Handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1
        # Write the m21 stream to a midi file
        stream.write(format, file_name)


# Create sample of seed
melody_generator = MelodyGenerator()
seed1 = "64 _ 69 _ _ _ 71 _ 72 _ _ 71 69 _ 76 _ _ _ _ _"
seed2 = "71 _ _ _ 74 _ 72 _ _ 71 69 _ 68 _ _ _ 69"

# Higher temperature == higher stochastic
melody = melody_generator.generate_melody(seed2, 1000, SEQUENCE_LENGTH, TEMPERATURE)
print(melody)
melody_generator.save_melody(melody)
