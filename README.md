# LSTM Melody Generator

This project is aimed at creating a deep learning model using Long Short-Term Memory (LSTM) for generating melodies. The model is trained on a [dataset](https://kern.humdrum.org/), specifically from the Alsace (eastern France) of MIDI files to learn the patterns and relationships in musical sequences.

![generated-music-piece](https://web.mit.edu/music21/doc/_images/what_18_0.png)

## Dataset
The dataset used on this project is from (Alsace)[https://kern.humdrum.org/cgi-bin/ksdata?l=essen/europa/elsass&format=zip], a historical region in France where they have plenty of collected compositions in `.krn` files that can be accessed using any music notation softwares.

## Installation
The following packages required to access and manipulate musical notations:
1. [MuseScore3](https://musescore.org/en)
2. [Music21](http://web.mit.edu/music21/)

### Installation
Start by cloning the repository to your local machine
`git clone https://github.com/<username>/lstm-melody-generator.git`

Set the MuseScore3 path to view the notations of `.krn` file sets
```
# set the default musicxmlPath file reader to MuseScore3.exe
us = m21.environment.UserSettings()
us["musicxmlPath"] = 'C:/Program Files/MuseScore 3/bin/MuseScore3.exe'
```

### Usage
1. Towards the preprocessing, provided are `.txt` files from the `.krn` file sets and are then tokenized stored within `mapping.json` as `file_dataset` being the final output of preprocessing
`python preprocess.py`

2. For training, an `.h5` file is produced for creating the melody
`python train.py`

To generate the model, run the following command; the seeds can be configured to the formatting required. 
`python melody_generator.py`

### Acknowledgments

[Valerio Velardo - The Sound of AI](https://www.youtube.com/@ValerioVelardoTheSoundofAI)
