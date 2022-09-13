from torch.utils.tensorboard import SummaryWriter
from plotting_utils import (
    plot_spectrogram_to_numpy, 
    plot_gate_outputs_to_numpy
)
import logging
import random
import torch


class TransformerTTSLogger(SummaryWriter):
    def __init__(self, logdir, terminal_log_file_path, hparams):
        super(TransformerTTSLogger, self).__init__(logdir)
        self.hparams = hparams
        self.terminal_log_file_path = terminal_log_file_path
        self.terminal_logger = self.logging_fn()
        self.terminal_logger.info(f"target step: {hparams.target_step}")
        self.terminal_logger.info(f"cuDNN Enabled: {hparams.cudnn_enabled}")
        self.terminal_logger.info(f"cuDNN Benchmark: {hparams.cudnn_benchmark}")
        self.terminal_logger.info(
            self.hparams
        )

    def log_training(self, loss, grad_norm, learning_rate, duration,
                     iteration):
        self.add_scalar("training.loss", loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)
        self.terminal_logger.info(
            "TrainLoss {} {:.6f} GradNorm {:.6f} LeaningRate {:.8f} {:.2f}s/it".format(
                iteration, loss, grad_norm, learning_rate, duration
            )
        )

    def log_validation(self, loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", loss, iteration)
        mel_outputs, _, gate_outputs = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        idx = random.randint(0, mel_outputs.size(0) - 1)
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.terminal_logger.info(
            "Validationloss {}: {:6f}  ".format(
                iteration, loss
            )
        )

    def logging_fn(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(self.terminal_log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger
