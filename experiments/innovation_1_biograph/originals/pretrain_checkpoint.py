"""Epoch-boundary persistence for the encoder-only V2 pretraining run."""
import os
import random
import tempfile

import numpy as np
import torch


def atomic_save(payload, path, keep_previous=False):
    """Commit a complete file; a failed write leaves the latest save intact."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + os.path.basename(path) + '.',
                                     suffix='.tmp', dir=directory)
    previous_temp = None
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if keep_previous and os.path.exists(path):
            previous_temp = temporary + '.previous'
            os.link(path, previous_temp)
            os.replace(previous_temp, path + '.previous')
            previous_temp = None
        os.replace(temporary, path)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        for leftover in (temporary, previous_temp):
            if leftover is not None and os.path.exists(leftover):
                os.unlink(leftover)


def save_training_checkpoint(path, model, optimizer, scheduler, epoch,
                             train_sample, validation_sample, config):
    payload = {
        'format': 'hyper_encoder_only_v2_pretraining_v1',
        'config': config,
        'completed_epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        # The next epoch shuffles the previous epoch's order, not df_train.
        'train_sample': train_sample,
        'validation_sample': validation_sample,
        'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    atomic_save(payload, path, keep_previous=True)
    print('Saved resumable pretraining checkpoint: epoch={} path={}'.format(
        epoch, path), flush=True)


def load_training_checkpoint(path, model, optimizer, scheduler, config):
    # These files are generated locally by this training pipeline.
    state = torch.load(path, map_location='cpu')
    if state.get('format') != 'hyper_encoder_only_v2_pretraining_v1':
        raise ValueError('Not a V2 pretraining resume checkpoint: ' + path)
    if state['config'] != config:
        raise ValueError('Checkpoint training settings differ from this run')
    epoch = state['completed_epoch']
    if not isinstance(epoch, int) or not 0 <= epoch <= config['num_epochs']:
        raise ValueError('Invalid completed epoch in checkpoint')
    if len(state['cuda_rng']) != torch.cuda.device_count():
        raise ValueError('CUDA device count differs from saved RNG state')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    print('Resuming pretraining after completed epoch {}'.format(epoch), flush=True)
    return epoch, state['train_sample'], state['validation_sample']
