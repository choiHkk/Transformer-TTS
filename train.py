import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from data_utils import (
    segmentation_train_and_validation_text_file,
    TextMelLoader,
    TextMelCollate,
)
from optimizer import ScheduledOptim
from torch.utils.data import DataLoader
from loss_function import TransformerTTSLoss
from logger import TransformerTTSLogger
from hparams import Hparams
import numpy as np
import argparse
import shutil
import random
import torch
import time
import os


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate()
    train_loader = DataLoader(
        trainset, num_workers=8, shuffle=True,
        sampler=None, batch_size=hparams.batch_size,
        pin_memory=True, drop_last=True, collate_fn=collate_fn
    )
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(hparams, output_directory, log_directory,
                                   initialize=False):
    log_dir = os.path.join(output_directory, log_directory)
    model_state_dir_name = os.path.join(
        log_dir,
        "model_state"
    )
    tensorboard_log_dir_name = os.path.join(
        log_dir,
        "runs"
    )
    terminal_log_file_path = os.path.join(
        log_dir,
        "Terminal_train_log"
    )

    if initialize:
        print("Initializing all directories")
        if os.path.isdir(log_dir):
            if os.path.isdir(model_state_dir_name):
                shutil.rmtree(model_state_dir_name)
            if os.path.isdir(tensorboard_log_dir_name):
                shutil.rmtree(tensorboard_log_dir_name)
            if os.path.isfile(terminal_log_file_path):
                os.remove(terminal_log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    if not os.path.isdir(model_state_dir_name):
        os.makedirs(model_state_dir_name)
        os.chmod(model_state_dir_name, 0o775)
    logger = TransformerTTSLogger(tensorboard_log_dir_name, terminal_log_file_path, hparams)
    return logger, model_state_dir_name


def load_model(hparams, gpu):
    from model import TransformerTTS
    model = TransformerTTS(hparams).cuda(gpu)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate,
                    iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer._optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(gpu, model, criterion, valset, iteration, batch_size,
             collate_fn, logger):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(valset, sampler=None, num_workers=8,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=True, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(*x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

    model.train()
    logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(gpu_nums, output_directory, log_directory, checkpoint_path,
          initialize, hparams):
    """
    Training and validation logging results to tensorboard and stdout
    """
    main_gpu = gpu_nums[0]

    model = load_model(hparams, main_gpu)
    if len(gpu_nums) > 1:
        model = torch.nn.DataParallel(
            model, device_ids=gpu_nums, output_device=main_gpu)

    learning_rate = hparams.learning_rate
    optimizer = ScheduledOptim(model, hparams)
    criterion = TransformerTTSLoss().cuda(main_gpu)

    logger, model_state_dir_name = prepare_directories_and_logger(
        hparams, output_directory, log_directory, initialize)

    segmentation_train_and_validation_text_file(hparams)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    init_iteration = 0
    if checkpoint_path is not None:
        model, optimizer, _learning_rate, init_iteration = load_checkpoint(
            checkpoint_path, model, optimizer)
        if hparams.use_saved_learning_rate:
            learning_rate = _learning_rate

    iteration = init_iteration
    optimizer.current_step = iteration

    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    while True:
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(*x)

            loss = criterion(y_pred, y)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                hparams.grad_clip_thresh
            )

            optimizer.step_and_update_lr()
            cur_lr = optimizer._optimizer.param_groups[0]['lr']

            duration = time.perf_counter() - start
            logger.log_training(
                loss.item(), grad_norm, cur_lr, duration, iteration)

            if (iteration == init_iteration) or (iteration % hparams.save_per_step == 0):
                validate(main_gpu, model, criterion, valset, iteration,
                         hparams.batch_size, collate_fn, logger)

                checkpoint_path = os.path.join(
                    model_state_dir_name,
                    f"checkpoint_{iteration}"
                )
                save_checkpoint(model, optimizer, learning_rate, iteration,
                                checkpoint_path)

            # early stop training (excape from for loop)
            if (iteration == init_iteration+hparams.target_step):
                checkpoint_path = os.path.join(
                    model_state_dir_name,
                    f"checkpoint_{iteration}"
                )
                save_checkpoint(model, optimizer, learning_rate, iteration,
                                checkpoint_path)
                break

            iteration += 1

        # early stop training (excape from for loop)
        if (iteration == init_iteration+hparams.target_step):
            break


def synthesizer_train(args):
    """
    ==================================================================================================================================================

    main model training:
        python train.py -g 0 -o data -l kss_v1 -d ./kss

    ==================================================================================================================================================
    """
    hparams = Hparams
    hparams.data_path = args.data_path
    
    seed = hparams.seed
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU

    total_gpu_nums = list(map(str, list(range(torch.cuda.device_count()))))

    if args.gpu_nums is not None:
        gpu_nums = list(map(str, args.gpu_nums.split(",")))
        gpu_nums = list(set(total_gpu_nums).intersection(gpu_nums))
    os.environ["CUDA_VISIBLE_DEVICES"]= ",".join(list(map(str, gpu_nums)))
    gpu_nums = list(map(int, gpu_nums))

    train(gpu_nums, args.output_directory, args.log_directory,
          args.checkpoint_path, args.initialize, hparams)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_nums", type=str, default="0",
                        required=False)
    parser.add_argument('-o', '--output_directory', type=str,
                        required=False, default='data', help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        required=True, help='log files directory path')
    parser.add_argument('-d', '--data_path', type=str,
                        required=True, help='dataset path')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('-i', '--initialize', type=str2bool, default=False,
                        required=False, help='initialize log directory')
    args = parser.parse_args()
    synthesizer_train(args)
