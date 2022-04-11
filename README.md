
<h2 align='center'> 
    Realt-Time State-of-the-Art Speech Synthesis in Tensorflow v2 for Kiswahili Language
</h2>

This repository is an implementation of Text To Speech and has been possible due to original work done on Text To Speech in English and other Languages like German and Korean, which can be found [here.](https://github.com/TensorSpeech/TensorFlowTTS)

# Requirements
- Python 3.8.3
- Tensorflow 2.2
- cuda 11.4
- cudaDNN 8.2
- Tensorflow Addons >= 0.10.0

# Data Collection
This project was implemented on Dataset obtained from the [Word Project](https://www.wordproject.org/). Its a collection of all Bible books from Genesis to Revelation. Each of the book is written chapter-wise, and a corresposnding audio for the words written.
A total of 7108 sentences was obtained, ranging for 5 different books.


# Data Preprocessing  
For data processing, the format followed was like the [ljspeech dataset](https://keithito.com/LJ-Speech-Dataset/).

The audio files were programatically split into short audio clips based on silence. They were then combined based on a random length such that each of eventual audio file lies between 1 to 10 seconds as its for the ljspeech dataset, but still maintaining the order in which they followed each other.
This was done using python 3. The audio files were saved as a single channel,16 PCM WAVE file with a sampling rate of 22.05 kHz

The texts for each of the audio files were then manually mapped by listening to each of the audio file. At the end, a total of 7108 lines of sentences were obtained, corresponding to the audio clips.

For the individual sentences, the format is id, text and normalised text each separated by a pipe
```
    id | text | normalized text

    e.g.
    Kiswa-00001 | Mithali za Sulemani,Mlango 1, Mithali za Suleimani | Mithali za Sulemani,Mlango wa kwanza, Mitahli za Suleimani

```
An example audio clip, it's for the above text, is below:


<figure>
    <figcaption>Listen to Kiswa-00001.wav:</figcaption>
    <audio
        controls
        src="./Kiswa-00001.mp3">
            Your browser does not support the
            <code>audio</code> element.
    </audio>
</figure>

In case your browser does not support this audio format, download it here [Kiswa-00001.wav](https://github.com/Rono8/kiswahili_tts/blob/master/Kiswa-00001.wav)

# Background Research
When doing research on Language Processing, an interesting concept that is important is the **Arpabets**, the original TTS was done on English text, but because we want to do it for Kiswahili we will need to use the Kisahili  Arpabets. Below is the list of Kiswahili Arpabets

```
    'A', 'E', 'I', 'O', 'U', 
    'B', 'CH', 'D', "DH", 'F', 'G', 'GH', 'H',  'J', 'K', 'KH', 'L', 'M',
    'N','NG',"NG'", "NY", 'P', 'R', 'S', 'SH', 'T', "TH", 'V',  'W', 'Y', 'Z', 

```
Another important point to consider is the normalization, for instance in English a word like Eng. Is expanded to Engineer.
Below are the symbols and there normalised forms used in this research

```
    ('Bw.', 'Bwana'),
    ('Bi.', 'Bibi'), 
    ('sh.', 'shilingi'),
    ('Dkt', 'Daktari'), 
    ('–', 'hadi'),
    ('prof' , 'profesa'),
    ('n.k.' , 'na kadhalika'),
    ('pst', 'pasta')
```
# Installation
First clone the Tensorflow TTS repo
```
    git clone https://github.com/TensorSpeech/TensorFlowTTS.git

```
chnage into the directory where your code has been cloned to.

A note should be made, **do not** perform **the next ** step before working on your code to change the arpabets and normalised words as documented ealier in this instructions.

When  done, then install the rest of the requirements by the command. 

```
pip install . 

or

python setup.py install

```
The earlier warning is because the python script 'setup.py' generates the module which will be used later to preprocess audio and text into numpy arrays, now the module once created using the initial run generates arpabets and normalised words which are not changed easily later on no matter how many times you run the script again. 

## Preprocess and Normalize

After making sure the dataset is in the format line that of ljspeech, like below, in a subfolder called data in the root of the project directory;
```
|- ljspeech/
|   |- metadata.csv
|   |- wavs/
|       |- Kiswa-00001.wav
|       |- ...
```
The final prepeared data ready to be preprocessed can be donwloaded [here.](https://data.mendeley.com/datasets/vbvj6j6pm9/1)

You can now run the below commands to preprocess your data, compute statistics and normalize.

```
tensorflow-tts-preprocess --rootdir ./data/LJSpeech-1.1/ --outdir ./data/preprocess_dir/dump/ --conf ./preprocess/ljspeech_preprocess.yaml --dataset ljspeech

tensorflow-tts-normalize --rootdir ./data/LJSpeech-1.1/ --outdir ./data/preprocess_dir/dump/  --conf ./preprocess/ljspeech_preprocess.yaml --dataset ljspeech
```

The arguments to be input in the command prompt or powershell means the following;
- --rootdir , this is the root directory where your originla dataset is.
- --outdir, the directory where the preprocessed, computed and normalised data is stored
- --config, this is the path to a yml file containing configuration paramenters for preprocessing the data



# Model Training
Below is the exact python script used during the training of the model, after implementing the Abstract and Trainer Classes.

```
def start_training(train_dir, dev_dir, outdir, config,   use_norm, mixed_precision, resume, CUDA_VISIBLE_DEVICES, verbose):
        """Run training process."""

        # return strategy
        STRATEGY = return_strategy()

        # set mixed precision config
        if mixed_precision == 1:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

        mixed_precision = bool(mixed_precision)
        use_norm = bool(use_norm)

        # set logger
        if verbose > 1:
            logging.basicConfig(
                level=logging.DEBUG,
                stream=sys.stdout,
                format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )
        elif verbose > 0:
            logging.basicConfig(
                level=logging.INFO,
                stream=sys.stdout,
                format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )
        else:
            logging.basicConfig(
                level=logging.WARN,
                stream=sys.stdout,
                format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )
            logging.warning("Skip DEBUG/INFO messages")

        # check directory existence
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # check arguments
        if train_dir is None:
            raise ValueError("Please specify --train-dir")
        if dev_dir is None:
            raise ValueError("Please specify --valid-dir")

        # load and save config
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config = config
        config["version"] = tensorflow_tts.__version__

        # get dataset
        if config["remove_short_samples"]:
            mel_length_threshold = config["mel_length_threshold"]
        else:
            mel_length_threshold = 0

        if config["format"] == "npy":
            charactor_query = "*-ids.npy"
            mel_query = "*-raw-feats.npy" if use_norm is False else "*-norm-feats.npy"
            charactor_load_fn = np.load
            mel_load_fn = np.load
        else:
            raise ValueError("Only npy are supported.")

        train_dataset = CharactorMelDataset(
            dataset=config["tacotron2_params"]["dataset"],
            root_dir=train_dir,
            charactor_query=charactor_query,
            mel_query=mel_query,
            charactor_load_fn=charactor_load_fn,
            mel_load_fn=mel_load_fn,
            mel_length_threshold=mel_length_threshold,
            reduction_factor=config["tacotron2_params"]["reduction_factor"],
            use_fixed_shapes=config["use_fixed_shapes"],
        )

        # update max_mel_length and max_char_length to config
        config.update({"max_mel_length": int(train_dataset.max_mel_length)})
        config.update({"max_char_length": int(train_dataset.max_char_length)})

        with open(os.path.join(outdir, "config.yml"), "w") as f:
            yaml.dump(config, f, Dumper=yaml.Dumper)
        for key, value in config.items():
            logging.info(f"{key} = {value}")

        train_dataset = train_dataset.create(
            is_shuffle=config["is_shuffle"],
            allow_cache=config["allow_cache"],
            batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
        )

        valid_dataset = CharactorMelDataset(
            dataset=config["tacotron2_params"]["dataset"],
            root_dir=dev_dir,
            charactor_query=charactor_query,
            mel_query=mel_query,
            charactor_load_fn=charactor_load_fn,
            mel_load_fn=mel_load_fn,
            mel_length_threshold=mel_length_threshold,
            reduction_factor=config["tacotron2_params"]["reduction_factor"],
            use_fixed_shapes=False,  # don't need apply fixed shape for evaluation.
        ).create(
            is_shuffle=config["is_shuffle"],
            allow_cache=config["allow_cache"],
            batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
        )

        # define trainer

        config['outdir'] = outdir
        trainer = Tacotron2Trainer(
            config=config,
            strategy=STRATEGY,
            steps=0,
            epochs=0,
            is_mixed_precision=mixed_precision,
        )

        with STRATEGY.scope():
            # define model.
            tacotron_config = Tacotron2Config(**config["tacotron2_params"])
            tacotron2 = TFTacotron2(config=tacotron_config,  trainable=True, name="tacotron2")
            tacotron2._build()
            tacotron2.summary()
            
            # if len(pretrained) > 1:
            #     tacotron2.load_weights(pretrained, by_name=True, skip_mismatch=True)
            #     logging.info(f"Successfully loaded pretrained weight from {pretrained}.")

            # AdamW for tacotron2
            learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
                decay_steps=config["optimizer_params"]["decay_steps"],
                end_learning_rate=config["optimizer_params"]["end_learning_rate"],
            )

            learning_rate_fn = WarmUp(
                initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
                decay_schedule_fn=learning_rate_fn,
                warmup_steps=int(
                    config["train_max_steps"]
                    * config["optimizer_params"]["warmup_proportion"]
                ),
            )

            optimizer = AdamWeightDecay(
                learning_rate=learning_rate_fn,
                weight_decay_rate=config["optimizer_params"]["weight_decay"],
                beta_1=0.9,
                beta_2=0.98,
                epsilon=1e-6,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            )

            _ = optimizer.iterations

        # compile trainer
        trainer.compile(model=tacotron2, optimizer=optimizer)

        # start training
        try:
            trainer.fit(
                train_dataset,
                valid_dataset,
                saved_path=os.path.join(config["outdir"], "checkpoints/"),
                resume=resume,
            )
        except KeyboardInterrupt:
            trainer.save_checkpoint()
            logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")



train_dir = './data/preprocess_dir/dump/train/'
val_dir = "./data/preprocess_dir/dump/valid/"
outdir = './data/processed/'
config = './examples/tacotron2/conf/tacotron2.v1.yaml'
use_norm = 1
mixed_precision = 0
resume = "" 
#resume = './data/Processed/checkpoints/ckpt-4'
CUDA_VISIBLE_DEVICES = 0
verbose = 1


start_training(train_dir, val_dir, outdir, config, use_norm, mixed_precision, resume, CUDA_VISIBLE_DEVICES, verbose)
```
In cmd or ps, run the command below to start training

```
python .\tensorflow_tts\bin\preprocess.py --rootdir ./data/preprocess_dir/dump/ --outdir ./data/preprocess_dir/dump/ --conf ./preprocess/ljspeech_preprocess.yaml --dataset ljspeech          

```
if after sometime the training stops and you would want to resume, kindly use the command below

```

python ./examples/melgan/train_melgan.py --train-dir ./data/preprocess_dir/dump/train/ --dev-dir ./data/preprocess_dir/dump/valid/ --outdir ./examples/melgan/exp/train.melgan.v2/ --config ./examples/melgan/conf/melgan.v2.yaml --resume ""

```
# Results
For a test of the model that was trained on 150k iterations, download it from google drive [here](https://drive.google.com/file/d/1-LvxuDxfxJ8CyUurNkjeo_lXiQeXfuMY/view?usp=sharing) and test it on your own kiswahili sentence.
You could even perform more training or fine tuning the model to get optimal results and contribute to this work. Just make a pull request or contact via email through
the contact provided at the end of this readme file.

# Kiswahili TTS Synthesis
The system is available and can be accessed using the link below
https://colab.research.google.com/drive/17ZZKB54T1cdPB6j47wBDkADG8UMW1jQd

If you have followed all the above steps and would want to test this model and see its performance, you can use the code below:

Be careful to change to write paths depending on how you have named your folders, but its okay if you use exactly the way this repo is.
````
! git clone https://github.com/Rono8/kiswahili_tts
os.chdir("./kiswahili_tts")
!pip install .
!pip install git+https://github.com/repodiac/german_transliterate.git #egg=german_transliterate
!pip uninstall tensorflow -y
!pip install tensorflow==2.3
import itertools
import logging
import os
import random
import argparse
import logging

import numpy as np
import yaml
from tqdm import tqdm

import tensorflow as tf
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor

import IPython.display as ipd
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
# Download pretrained  model zip file from drive
print("Downloading zip model...")
!gdown --id {"1j7IHz_2m60vVhxgp9VK3EXd2JlKz2xk3"} -O model-150000.h5
#!gdown --id {"1-1cpmelBQG2VaptQk8bFYmrUyhMYyivL"} -O tacotron2.v1.yaml

# Download pretrained Vocoder model
print("Downloading MelGAN model...")
!gdown --id {"1A3zJwzlXEpu_jHeatlMdyPGjn1V7-9iG"} -O melgan-1M6.h5
!gdown --id {"1Ys-twSd3m2uqhJOEiobNox6RNQf4txZs"} -O melgan_config.yml


# Load Vocoder
melgan_config = AutoConfig.from_pretrained('/content/kiswahili_tts/examples/melgan/conf/melgan.v1.yaml')
melgan = TFAutoModel.from_pretrained(
    config=melgan_config,
   pretrained_path="melgan-1M6.h5",
    name="melgan"
)

# Load Model
tacotron2_config = AutoConfig.from_pretrained('/content/kiswahili_tts/examples/tacotron2/conf/tacotron2.v1.yaml')
tacotron2 = TFAutoModel.from_pretrained(
    config=tacotron2_config,
    
    pretrained_path="/content/kiswahili_tts/model-150000.h5",
    name="tacotron2"
)

#processor = AutoProcessor.from_pretrained(pretrained_path="/content/drive/MyDrive/ljspeech_mapper.json")
processor = AutoProcessor.from_pretrained(pretrained_path="/content/kiswahili_tts/tensorflow_tts/ljspeech_mapper.json")

def do_synthesis(input_text, text2mel_model, vocoder_model):
    input_ids = processor.text_to_sequence(input_text)

    # text2mel part
    _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        tf.convert_to_tensor([len(input_ids)], tf.int32),
        tf.convert_to_tensor([0], dtype=tf.int32)
    )

    # vocoder part
    audio = vocoder_model(mel_outputs)[0, :, 0]

    return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()

def visualize_attention(alignment_history):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title(f'Alignment steps')
    im = ax.imshow(
        alignment_history,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.show()
    plt.close()

def visualize_mel_spectrogram(mels):
    mels = tf.reshape(mels, [-1, 80]).numpy()
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(311)
    ax1.set_title(f'Predicted Mel-after-Spectrogram')
    im = ax1.imshow(np.rot90(mels), aspect='auto', interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
    plt.show()
    plt.close()

text = input('Input the Kiswahili text:')
print(text)
mels, alignment_history, audios = do_synthesis(text, tacotron2, melgan)
visualize_attention(alignment_history[0])
visualize_mel_spectrogram(mels[0])
ipd.Audio(audios, rate=22050)

````

# Common Errors and Fixes
Some of the errors encounted due to compatibility issues of modules, and their solutions are:

``
1. 
error - AttributeError: module 'typing' has no attribute '_ClassVar'
sol   - pip uninstall TensorflowTTS -y
        comment out the line with dataclassses in the requirements in setup.py
        python .\setup.py install
``


``
2. When training and the sytem crashes, kinldy lower the batch size to 8 or 4, note that at the hood computations are done in matrices, it become very big matrices when large batch sizes is used and this causes the system to crash due to deficiency of memory.
``
# Contact
For more information, kindly contact
kiptookelvin96@gmail.com 

# Licence
[Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) 

# Acknowledgement
Much thanks to the team that helped work on data processing, which included manually relating textfiles to their corresponding audio files.

I am also grateful to my research supervisors, Dr. Ciira Maina, Director of the Center for Data Science and Artificial Intelligence, and Prof Elijah Mwangi, Faculty of Engineering, University of Nairobi, who have offered significant assistance throughout this research project.
