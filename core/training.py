# face2eeg/core/training.py

"""
Single training loop with optional validation, checkpointing,
LR scheduling, and resume support.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

def train_model(
    model,
    X_train, y_train,
    X_val=None, y_val=None,
    device=None,
    epochs=30,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=1e-5,
    loss_fn=None,
    scheduler_step_size=5,
    scheduler_gamma=0.1,
    scheduler=None,
    optimizer=None,
    model_save_path=None,
    resume_from=None,
    verbose=True,
):
    """
    Train a model with optional validation and full resume support.

    Parameters
    ----------
    model : nn.Module
    X_train, y_train : np.ndarray
        Training data.
    X_val, y_val : np.ndarray or None
        Validation data. If None, validation is skipped.
    device : torch.device or None
        If None, auto-detects CUDA.
    epochs : int
        Total number of epochs (absolute, not relative to start_epoch).
    batch_size : int
    learning_rate, weight_decay : float
        Used only when `optimizer` is None.
    loss_fn : callable or None
        Default: nn.MSELoss().
    scheduler_step_size, scheduler_gamma : int, float
        Used only when `scheduler` is None (StepLR config).
    scheduler, optimizer : torch objects or None
        Custom instances. When resuming, their state_dicts will be
        overwritten from the checkpoint.
    model_save_path : str or None
        Where to write checkpoints. If None, no saving.
        The checkpoint is overwritten every epoch.
    resume_from : str or None
        Path to a checkpoint. If provided and file exists, training
        resumes from `ckpt['epoch']` with restored optimizer/scheduler
        and history.
    verbose : bool

    Returns
    -------
    history : dict
        {
            'loss'      : list[float],   # mean train loss per epoch
            'val_loss'  : list[float],   # mean val loss per epoch
            'lr'        : list[float],   # LR after scheduler.step(), per epoch
            'best_val'  : float,         # best val loss seen so far
            'best_epoch': int or None,   # 1-based epoch of best val
            'time'      : float,         # seconds spent in this call
        }
    """
    # ------------------------------------------------------------------ setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # cudnn autotuning for fixed input shapes
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    if loss_fn is None:
        loss_fn = nn.MSELoss()

    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    if scheduler is None:
        scheduler = StepLR(
            optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        )

    has_val = (X_val is not None) and (y_val is not None)

    # ------------------------------------------------------------- history
    history = {
        'loss': [],
        'val_loss': [],
        'lr': [],
        'best_val': float('inf'),
        'best_epoch': None,
        'time': 0.0,
    }
    start_epoch = 0

    # -------------------------------------------------------------- resume
    if resume_from is not None:
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

        ckpt = torch.load(resume_from, map_location=device)

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        start_epoch = int(ckpt['epoch'])
        history['loss']       = list(ckpt.get('loss', []))
        history['val_loss']   = list(ckpt.get('val_loss', []))
        history['lr']         = list(ckpt.get('lr', []))
        history['best_val']   = ckpt.get('best_val', float('inf'))
        history['best_epoch'] = ckpt.get('best_epoch', None)

        if verbose:
            print(
                f"[resume] '{resume_from}' "
                f"(epoch={start_epoch}, best_val={history['best_val']:.4f})"
            )

        if start_epoch >= epochs:
            if verbose:
                print(
                    f"[resume] already trained {start_epoch} epochs "
                    f">= requested {epochs}. Nothing to do."
                )
            return history

    # ---------------------------------------------- preload data
    # Convert once instead of per-batch. Stays on CPU; moved to device
    # per batch (X and y) to keep training dynamics identical.
    X_train_t = torch.from_numpy(np.ascontiguousarray(X_train)).float()
    y_train_t = torch.from_numpy(np.ascontiguousarray(y_train)).float()
    if has_val:
        X_val_t = torch.from_numpy(np.ascontiguousarray(X_val)).float()
        y_val_t = torch.from_numpy(np.ascontiguousarray(y_val)).float()

    N_train = X_train_t.shape[0]
    N_val = X_val_t.shape[0] if has_val else 0

    # ---------------------------------------------------------- train loop
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        # ----------------------------------------------- Training phase
        model.train()
        # accumulate on device, no per-batch .item() sync
        loss_acc = torch.zeros((), device=device)
        n_batches = 0

        for i in range(0, N_train, batch_size):
            # slicing preloaded tensors (no numpy→torch per batch)
            # move both X and y to device; drop .cpu() on y_pred
            X = X_train_t[i:i + batch_size].to(device, non_blocking=True)
            y = y_train_t[i:i + batch_size].to(device, non_blocking=True)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            # set_to_none=True (faster, same gradients)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_acc += loss.detach()
            n_batches += 1

        train_loss = (loss_acc / n_batches).item()   # one sync per epoch
        history['loss'].append(train_loss)

        # ---------------------------------------------- Validation phase
        if has_val:
            model.eval()
            # inference_mode via context manager
            with torch.inference_mode():
                val_acc = torch.zeros((), device=device)
                n_val_batches = 0
                for i in range(0, N_val, batch_size):
                    X = X_val_t[i:i + batch_size].to(device, non_blocking=True)
                    y = y_val_t[i:i + batch_size].to(device, non_blocking=True)
                    y_pred = model(X)
                    val_acc += loss_fn(y_pred, y).detach()
                    n_val_batches += 1
                val_loss = (val_acc / n_val_batches).item()

            history['val_loss'].append(val_loss)

            if val_loss < history['best_val']:
                history['best_val'] = val_loss
                history['best_epoch'] = epoch + 1

        # ------------------------------------------------- Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # ----------------------------------------------------- Verbose
        if verbose:
            msg = f"epoch {epoch + 1:>3}/{epochs} — train={train_loss:.4f}"
            if has_val:
                msg += f" — val={history['val_loss'][-1]:.4f}"
            msg += f" — lr={current_lr:.2e}"
            print(msg)

        # ---------------------------------------------------- Checkpoint
        if model_save_path:
            ckpt = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': history['loss'],
                'val_loss': history['val_loss'],
                'lr': history['lr'],
                'best_val': history['best_val'],
                'best_epoch': history['best_epoch'],
            }
            torch.save(ckpt, model_save_path)

    history['time'] = time.time() - start_time
    return history