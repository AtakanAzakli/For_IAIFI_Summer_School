import os
import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import pandas as pd
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from sklearn.model_selection import train_test_split
# from tqdm import tqdm  # for progress tracking
from dbtransformer import DBTransformer

import src

pd.set_option('display.max_columns', None)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("GPU:", gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

try:
    EXPERIMENT_NAME = os.path.splitext(os.path.basename(__file__))[0]
except NameError:
    EXPERIMENT_NAME = "placeholder_name"

cfg = src.configs.get_configs()
cfg.experiment.name = EXPERIMENT_NAME
print(f' Experiment name: {cfg.experiment.name} '.center(80, "="))
MONITOR = "val_loss" if cfg.validation_split > 0.0 else "loss"

# =============================================================================
# =============================================================================
print(f' Changes made to the default config are: '.center(80, "-"))
cfg.batch_size = 128
cfg.epochs = 100
cfg.validation_split = 0.05

print(f' - batch_size: {cfg.batch_size}')
print(f' - epochs: {cfg.epochs}')
print(f' - validation_split: {cfg.validation_split}')
# =============================================================================
# =============================================================================


cfg_dict = OmegaConf.to_container(cfg, resolve=True)
cfg_dict["monitor"] = MONITOR

if cfg.experiment.wandb:
    wandb.init(project=cfg.experiment.name, config=cfg_dict)

print(cfg_dict)
print(cfg.save_pth.plots)

if cfg.experiment.save:
    if not os.path.exists(cfg.save_pth.model_save):
        os.makedirs(cfg.save_pth.model_save)
    if not os.path.exists(cfg.save_pth.plots):
        os.makedirs(cfg.save_pth.plots)
    if not os.path.exists(cfg.save_pth.checkpoint):
        os.makedirs(cfg.save_pth.checkpoint)


X_train, X_test, y_train, y_test = train_test_split(all_data, labels, test_size=0.2, random_state=cfg.seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42)

def random_permute_sequence(x, y):
    sequence_length = tf.shape(x)[0]  # T
    perm_indices = tf.random.shuffle(tf.range(sequence_length))
    x_permuted = tf.gather(x, perm_indices, axis=0)
    return x_permuted, y

batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

train_dataset = (
    train_dataset
    .shuffle(1000)  # Shuffle the sample order
    .map(random_permute_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)


callbacks = []
if cfg.early_stopping:
    callbacks.append(tf.keras.callbacks.EarlyStopping(MONITOR, patience=cfg.patience, min_delta=cfg.min_delta, restore_best_weights=True))
if cfg.reduce_lr_on_plateau:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, patience=int(cfg.patience/2), min_delta=0.0, min_lr=cfg.min_lr))
if cfg.experiment.save:
    callbacks.append(EpochModelCheckpoint(filepath=cfg.save_pth.checkpoint, monitor=MONITOR, mode="min", save_best_only=True, save_weights_only=False, period=cfg.checkpoint_period))
if cfg.experiment.wandb:
    callbacks.append(WandbMetricsLogger())
    callbacks.append(WandbModelCheckpoint(filepath=f"{cfg.save_pth.checkpoint}/model.epoch{{epoch:02d}}.keras"))
print(callbacks)



model = DBTransformer(data_shape=X_train.shape)

optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy())
history = model.fit(X_train, y_train, batch_size=cfg.batch_size, shuffle=True, epochs=cfg.epochs, verbose=2, validation_split=cfg.validation_split, callbacks=callbacks)

predictions = model.predict(X_test, batch_size=1024)

# Compute metrics
from utils import compute_precision_recall, plot_roc_curve, plot_training_loss, plot_confusion_matrix

precision, recall = compute_precision_recall(model, X_test, y_test, cfg)
plot_roc_curve(model, X_test, y_test, cfg)
plot_training_loss(history, cfg)
plot_confusion_matrix(model, X_test, y_test, cfg)

if cfg.experiment.wandb:
    wandb.finish()

