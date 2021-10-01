import os
import sys

os.chdir("./kiswahili_tts/TensorFlowTTS")

import itertools
import logging
import random
import argparse
import numpy as np
import yaml
import tensorflow as tf
import tensorflow_tts

from tqdm import tqdm
from tensorflow_tts.datasets.abstract_dataset import AbstractDataset
from tensorflow_tts.utils import find_files
from tensorflow_tts.configs.tacotron2 import Tacotron2Config
from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.optimizers import AdamWeightDecay, WarmUp
from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.utils import (calculate_2d_loss, calculate_3d_loss,
                                  return_strategy)



physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)



class CharactorMelDataset(AbstractDataset):
    """Tensorflow Charactor Mel dataset."""

    def __init__(
        self,
        dataset,
        root_dir,
        charactor_query="*-ids.npy",
        mel_query="*-norm-feats.npy",
        charactor_load_fn=np.load,
        mel_load_fn=np.load,
        mel_length_threshold=0,
        reduction_factor=1,
        mel_pad_value=0.0,
        char_pad_value=0,
        ga_pad_value=-1.0,
        g=0.2,
        use_fixed_shapes=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            charactor_query (str): Query to find charactor files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            charactor_load_fn (func): Function to load charactor file.
            mel_load_fn (func): Function to load feature file.
            mel_length_threshold (int): Threshold to remove short feature files.
            reduction_factor (int): Reduction factor on Tacotron-2 paper.
            mel_pad_value (float): Padding value for mel-spectrogram.
            char_pad_value (int): Padding value for charactor.
            ga_pad_value (float): Padding value for guided attention.
            g (float): G value for guided attention.
            use_fixed_shapes (bool): Use fixed shape for mel targets or not.
            max_char_length (int): maximum charactor length if use_fixed_shapes=True.
            max_mel_length (int): maximum mel length if use_fixed_shapes=True

        """
        # find all of charactor and mel files.
        charactor_files = sorted(find_files(root_dir, charactor_query))
        mel_files = sorted(find_files(root_dir, mel_query))
        mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
        char_lengths = [charactor_load_fn(f).shape[0] for f in charactor_files]

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mels files in ${root_dir}."
        assert (
            len(mel_files) == len(charactor_files) == len(mel_lengths)
        ), f"Number of charactor, mel and duration files are different \
                ({len(mel_files)} vs {len(charactor_files)} vs {len(mel_lengths)})."

        if ".npy" in charactor_query:
            suffix = charactor_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in charactor_files]

        # set global params
        self.utt_ids = utt_ids
        self.mel_files = mel_files
        self.charactor_files = charactor_files
        self.mel_load_fn = mel_load_fn
        self.charactor_load_fn = charactor_load_fn
        self.mel_lengths = mel_lengths
        self.char_lengths = char_lengths
        self.reduction_factor = reduction_factor
        self.mel_length_threshold = mel_length_threshold
        self.mel_pad_value = mel_pad_value
        self.char_pad_value = char_pad_value
        self.ga_pad_value = ga_pad_value
        self.g = g
        self.use_fixed_shapes = use_fixed_shapes
        self.max_char_length = np.max(char_lengths)

        if np.max(mel_lengths) % self.reduction_factor == 0:
            self.max_mel_length = np.max(mel_lengths)
        else:
            self.max_mel_length = (
                np.max(mel_lengths)
                + self.reduction_factor
                - np.max(mel_lengths) % self.reduction_factor
            )

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            mel_file = self.mel_files[i]
            charactor_file = self.charactor_files[i]

            items = {
                "utt_ids": utt_id,
                "mel_files": mel_file,
                "charactor_files": charactor_file,
            }

            yield items
    
    @tf.function
    def _load_data(self, items):
        mel = tf.numpy_function(np.load, [items["mel_files"]], tf.float32)
        charactor = tf.numpy_function(np.load, [items["charactor_files"]], tf.int32)
        mel_length = len(mel)
        char_length = len(charactor)
        # padding mel to make its length is multiple of reduction factor.
        real_mel_length = mel_length
        remainder = mel_length % self.reduction_factor
        if remainder != 0:
            new_mel_length = mel_length + self.reduction_factor - remainder
            mel = tf.pad(
                mel,
                [[0, new_mel_length - mel_length], [0, 0]],
                constant_values=self.mel_pad_value,
            )
            mel_length = new_mel_length

        items = {
            "utt_ids": items["utt_ids"],
            "input_ids": charactor,
            "input_lengths": char_length,
            "speaker_ids": 0,
            "mel_gts": mel,
            "mel_lengths": mel_length,
            "real_mel_lengths": real_mel_length,
        }

        return items

    def _guided_attention(self, items):
        """Guided attention. Refer to page 3 on the paper (https://arxiv.org/abs/1710.08969)."""
        items = items.copy()
        mel_len = items["mel_lengths"] // self.reduction_factor
        char_len = items["input_lengths"]
        xv, yv = tf.meshgrid(tf.range(char_len), tf.range(mel_len), indexing="ij")
        f32_matrix = tf.cast(yv / mel_len - xv / char_len, tf.float32)
        items["g_attentions"] = 1.0 - tf.math.exp(
            -(f32_matrix ** 2) / (2 * self.g ** 2)
        )
        return items

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
    ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        datasets = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, args=(self.get_args())
        )

        # load data
        datasets = datasets.map(
            lambda items: self._load_data(items),
            tf.data.experimental.AUTOTUNE
        )

        # calculate guided attention
        datasets = datasets.map(
            lambda items: self._guided_attention(items),
            tf.data.experimental.AUTOTUNE
        )

        datasets = datasets.filter(
            lambda x: x["mel_lengths"] > self.mel_length_threshold
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        # define padding value.
        padding_values = {
            "utt_ids": " ",
            "input_ids": self.char_pad_value,
            "input_lengths": 0,
            "speaker_ids": 0,
            "mel_gts": self.mel_pad_value,
            "mel_lengths": 0,
            "real_mel_lengths": 0,
            "g_attentions": self.ga_pad_value,
        }

        # define padded shapes.
        padded_shapes = {
            "utt_ids": [],
            "input_ids": [None]
            if self.use_fixed_shapes is False
            else [self.max_char_length],
            "input_lengths": [],
            "speaker_ids": [],
            "mel_gts": [None, 80]
            if self.use_fixed_shapes is False
            else [self.max_mel_length, 80],
            "mel_lengths": [],
            "real_mel_lengths": [],
            "g_attentions": [None, None]
            if self.use_fixed_shapes is False
            else [self.max_char_length, self.max_mel_length // self.reduction_factor],
        }

        datasets = datasets.padded_batch(
            batch_size, padded_shapes=padded_shapes, padding_values=padding_values
        )
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = {
            "utt_ids": tf.string,
            "mel_files": tf.string,
            "charactor_files": tf.string,
        }
        return output_types

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "CharactorMelDataset"

    
# Trainer Class
class Tacotron2Trainer(Seq2SeqBasedTrainer):
    """Tacotron2 Trainer class based on Seq2SeqBasedTrainer."""

    def __init__(
        self, config, strategy, steps=0, epochs=0, is_mixed_precision=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_mixed_precision (bool): Use mixed precision or not.

        """
        super(Tacotron2Trainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            strategy=strategy,
            is_mixed_precision=is_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "stop_token_loss",
            "mel_loss_before",
            "mel_loss_after",
            "guided_attention_loss",
        ]
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

        self.config = config

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def _train_step(self, batch):
        """Here we re-define _train_step because apply input_signature make
        the training progress slower on my experiment. Note that input_signature
        is apply on based_trainer by default.
        """
        if self._already_apply_input_signature is False:
            self.one_step_forward = tf.function(
                self._one_step_forward, experimental_relax_shapes=True
            )
            self.one_step_evaluate = tf.function(
                self._one_step_evaluate, experimental_relax_shapes=True
            )
            self.one_step_predict = tf.function(
                self._one_step_predict, experimental_relax_shapes=True
            )
            self._already_apply_input_signature = True

        # run one_step_forward
        self.one_step_forward(batch)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def compute_per_example_losses(self, batch, outputs):
        """Compute per example losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        (
            decoder_output,
            post_mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = outputs

        mel_loss_before = calculate_3d_loss(
            batch["mel_gts"], decoder_output, loss_fn=self.mae
        )
        mel_loss_after = calculate_3d_loss(
            batch["mel_gts"], post_mel_outputs, loss_fn=self.mae
        )

        # calculate stop_loss
        max_mel_length = (
            tf.reduce_max(batch["mel_lengths"])
            if self.config["use_fixed_shapes"] is False
            else [self.config["max_mel_length"]]
        )
        stop_gts = tf.expand_dims(
            tf.range(tf.reduce_max(max_mel_length), dtype=tf.int32), 0
        )  # [1, max_len]
        stop_gts = tf.tile(
            stop_gts, [tf.shape(batch["mel_lengths"])[0], 1]
        )  # [B, max_len]
        stop_gts = tf.cast(
            tf.math.greater_equal(stop_gts, tf.expand_dims(batch["mel_lengths"], 1)),
            tf.float32,
        )

        stop_token_loss = calculate_2d_loss(
            stop_gts, stop_token_predictions, loss_fn=self.binary_crossentropy
        )

        # calculate guided attention loss.
        attention_masks = tf.cast(
            tf.math.not_equal(batch["g_attentions"], -1.0), tf.float32
        )
        loss_att = tf.reduce_sum(
            tf.abs(alignment_historys * batch["g_attentions"]) * attention_masks,
            axis=[1, 2],
        )
        loss_att /= tf.reduce_sum(attention_masks, axis=[1, 2])

        per_example_losses = (
            stop_token_loss + mel_loss_before + mel_loss_after + loss_att
        )

        dict_metrics_losses = {
            "stop_token_loss": stop_token_loss,
            "mel_loss_before": mel_loss_before,
            "mel_loss_after": mel_loss_after,
            "guided_attention_loss": loss_att,
        }

        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # predict with tf.function for faster.
        outputs = self.one_step_predict(batch)
        (
            decoder_output,
            mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = outputs
        mel_gts = batch["mel_gts"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            mels_before = decoder_output.values[0].numpy()
            mels_after = mel_outputs.values[0].numpy()
            mel_gts = mel_gts.values[0].numpy()
            alignment_historys = alignment_historys.values[0].numpy()
        except Exception:
            mels_before = decoder_output.numpy()
            mels_after = mel_outputs.numpy()
            mel_gts = mel_gts.numpy()
            alignment_historys = alignment_historys.numpy()

        # check directory
        utt_ids = batch["utt_ids"].numpy()
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (mel_gt, mel_before, mel_after, alignment_history) in enumerate(
            zip(mel_gts, mels_before, mels_after, alignment_historys), 0
        ):
            mel_gt = tf.reshape(mel_gt, (-1, 80)).numpy()  # [length, 80]
            mel_before = tf.reshape(mel_before, (-1, 80)).numpy()  # [length, 80]
            mel_after = tf.reshape(mel_after, (-1, 80)).numpy()  # [length, 80]

            # plot figure and save it
            utt_id = utt_ids[idx]
            figname = os.path.join(dirname, f"{utt_id}.png")
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            im = ax1.imshow(np.rot90(mel_gt), aspect="auto", interpolation="none")
            ax1.set_title("Target Mel-Spectrogram")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
            ax2.set_title(f"Predicted Mel-before-Spectrogram @ {self.steps} steps")
            im = ax2.imshow(np.rot90(mel_before), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
            ax3.set_title(f"Predicted Mel-after-Spectrogram @ {self.steps} steps")
            im = ax3.imshow(np.rot90(mel_after), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax3)
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # plot alignment
            figname = os.path.join(dirname, f"{idx}_alignment.png")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_title(f"Alignment @ {self.steps} steps")
            im = ax.imshow(
                alignment_history, aspect="auto", origin="lower", interpolation="none"
            )
            fig.colorbar(im, ax=ax)
            xlabel = "Decoder timestep"
            plt.xlabel(xlabel)
            plt.ylabel("Encoder timestep")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()



def start_training(train_dir, dev_dir, outdir, config, use_norm, mixed_precision, resume, CUDA_VISIBLE_DEVICES, verbose):
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