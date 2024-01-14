import os
import json
import numpy as np
import music21 as m21
import tensorflow.keras as keras
from constants import RESOURCE_PATH, KERN_DATASET_PATH, SAVE_DIR, MAPPING_PATH, SINGLE_FILE_DATASET, SEQUENCE_LENGTH, MUSIC_XML_PATH

# Note values
ACCEPTABLE_DURATIONS = [
    0.25, # sixteenth
    0.5, # eighth
    0.75, # dotted eighth
    1.0, # quarter
    1.5, # dotted quarter 
    2.0, # half
    3.0, # dotted half
    4.0 # whole
]

# Circle of Fifths
COF = {
    'C_Am': ['C', 'A'],
    'G_Em': ['G', 'E'],
    'D_Bm': ['D', 'B'],
    'A_F#m': ['A', 'F#'],
    'E_C#m': ['E', 'C#'],
    'B_G#m': ['B', 'G#'],
    'F#_D#m': ['F#', 'D#'],
    'C#_A#m': ['C#', 'A#'],
    'Ab_Fm': ['Ab', 'F'],
    'Eb_Cm': ['Eb', 'C'],
    'Bb_Gm': ['Bb', 'G'],
    'F_Dm': ['F', 'D'],
}


def initialize_client():
    us = m21.environment.UserSettings()
    try:
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        # set the default musicxmlPath file reader to MuseScore3.exe
        us["musicxmlPath"] = MUSIC_XML_PATH
    except Exception as e:
        print(f'Exception raised: {e}')


# kern, MIDI, MusicXML -> m21 -> kern, MIDI, ...
def load_songs_in_kern(dataset_path):
    songs = []

    try:
        # go through all the files in the dataset and load them with music21
        for path, subdir, files in os.walk(dataset_path):
            for file in files:
                if file[-3:] == "krn":
                    song = m21.converter.parse(os.path.join(path, file))
                    songs.append(song)

        return songs
    except Exception as e:
        print(f'Exception raised: {e}')


def has_acceptable_durations(song, acceptable_durations):
    for note in song.flatten().notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    # Get key of song
    global interval
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # Estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # Get interval for transposition. E.g., B major / G# minor -> C major / A minor
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch(COF['C_Am'][0]))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch(COF['C_Am'][1]))
        
    # Transpose song by calculated interval
    transposed_song = song.transpose(interval)
    return transposed_song


def encode_song(song, time_step=0.25):
    # p = 60 middle C, d = 1.0 quarter note -> [60, "_", "_", "_"]
    encoded_song = []
    for i, event in enumerate(song.flatten().notesAndRests):
        # Match if note
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi  # 60
        # Match if rest
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
            
        # Convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
        if i % 50 == 0 and i > 0:
            print(f'Encoding song index {i + 1} / {len(song.flatten().notesAndRests)}')

    # Cast encoded song to a str
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song


def preprocessing(dataset_path):
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    
    if not os.path.exists(RESOURCE_PATH):
        os.mkdir(RESOURCE_PATH)
    
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs")
    
    try:
        for i, song in enumerate(songs):
            # Check each notes in a song if acceptable or not
            if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
                continue
            
            # Key on C major / A minor
            song = transpose(song)

            # Encode songs with music time series representation
            encoded_song = encode_song(song)

            save_path = os.path.join(SAVE_DIR, f'song_{i}')
            with open(save_path, "w") as fp:
                fp.write(encoded_song)
            
            if i % 10 == 0 and i > 0:
                print(f'\nPreprocessing index {i + 1} / {len(songs)}...\n')
    except Exception as e:
        print(f'Exception raised: {e}')


def load(file_path):
    try:
        with open(file_path, "r") as fp:
            song = fp.read()
            return song
    except Exception as e:
        print(f'Exception raised: {e}')


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    try:
        new_song_delimiter = "/ " * sequence_length
        songs = ""
        # Load encoded songs and add delimiters
        for path, _, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(path, file)
                song = load(file_path)
                songs = songs + song + " " + new_song_delimiter
        songs = songs[:-1]

        # Save string that contains all datasets
        with open(file_dataset_path, "w") as fp:
            fp.write(songs)
        return songs
    except Exception as e:
        print(f'Exception raised: {e}')


def create_mapping(songs, mapping_path):
    try:
        mappings = {}

        # Identify vocabulary
        songs = songs.split()
        vocabulary = list(set(songs))

        # Create mappings
        for i, symbol in enumerate(vocabulary):
            mappings[symbol] = i

        # Save vocabulary to .json file
        with open(mapping_path, "w") as fp:
            json.dump(mappings, fp, indent=4)
        print(f'Successfully created mappings')
    except Exception as e:
        print(f'Failed creating mappingsnException raised: {e}')


def convert_songs_to_int(songs):
    try:
        int_songs = []
        # Load the mappings
        with open(MAPPING_PATH, "r") as fp:
            mappings = json.load(fp)

        # Cast songs to string to a list
        songs = songs.split()

        # Map songs to int
        for symbol in songs:
            int_songs.append(mappings[symbol])
        return int_songs
    except Exception as e:
        print(f'Exception raised: {e}')


def generate_training_sequences(sequence_length):
    try:
        # Load songs and map them to int
        songs = load(SINGLE_FILE_DATASET)
        int_songs = convert_songs_to_int(songs)

        # Generate the training sequences
        # 100 symbols, 64 sl, 100 - 64 = 36
        inputs = []
        targets = []

        num_sequences = len(int_songs) - sequence_length
        for i in range(num_sequences):
            # [11, 12, 13, 14, ...] -> i: [11, 12], t: 13; i: [12, 13], t: 14, ...
            inputs.append(int_songs[i: i + sequence_length])
            targets.append(int_songs[i + sequence_length])

        # One-hot encode the sequences
        # Inputs: (# of sequences, sequence length, vocabulary size)
        # [ [0, 1, 2], [1, 1, 2] ] -> [ [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ], [] ]
        vocabulary_size = len(set(int_songs))

        # Creates the input as a matrix
        inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
        targets = np.array(targets)
        print('Successfully created training sequences')
        return inputs, targets
    except Exception as e:
            print(f'Failed creating training sequences\nException raised: {e}')


def preprocess():
    initialize_client()
    preprocessing(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    print('Successfully preprocessed dataset')
    return inputs, targets
    

if __name__ == "__main__":
    inputs, targets = preprocess()
