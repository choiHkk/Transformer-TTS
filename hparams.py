# import tensorflow as tf
from text import symbols
import easydict


Hparams = easydict.EasyDict(
    ################################
    # Experiment Parameters        #
    ################################
    target_step=300000,
    save_per_step=1000,
    seed=1234,
    distributed_run=False,
    cudnn_enabled=True,
    cudnn_benchmark=False,
    ignore_layers=['embedding.weight'],

    ################################
    # Data Parameters             #
    ################################
    load_mel_from_disk=False,
    meta_files='kss/transcript.v.1.2.txt', 
    training_files='filelists/kss_audio_text_train_filelist.txt',
    validation_files='filelists/kss_audio_text_val_filelist.txt',
    # text_cleaners=['english_cleaners'],
    validation_num=64, 
    shuffle_num=3, 

    ################################
    # Audio Parameters             #
    ################################
    max_wav_value=32768.0,
    sampling_rate=22050,
    filter_length=1024,
    hop_length=256,
    win_length=1024,
    n_mel_channels=80,
    mel_fmin=0.0,
    mel_fmax=8000.0,
    
    preemphasize=True,
    preemphasis=0.97,
    ref_level_db=20,
    min_level_db=-100,
    signal_normalization=True,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
    use_lws=False,
    frame_shift_ms=None,
    max_abs_value=4.,

    ################################
    # Model Parameters             #
    ################################
    n_symbols=len(symbols),
    n_speakers=1, 
    d_model=256, 

    ################################
    # Optimization Hyperparameters #
    ################################
    use_saved_learning_rate=False,
    learning_rate=1e-3,
    weight_decay=1e-6,
    grad_clip_thresh=1.0,
    batch_size=32,
    fp16_run=True, 
    warm_up_step=4000, 
    anneal_steps=[300000, 400000, 500000], 
    anneal_rate=0.3, 
)
