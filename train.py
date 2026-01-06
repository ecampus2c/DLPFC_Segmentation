import os
import glob
import json
import numpy as np
import tensorflow as tf
from src.config import *
from src.data_loader import build_tf_dataset
from src.model import build_lightweight_unet
from src.utils import bce_dice_loss, dice_coef

def run_training():
    # 1. Find Data
    files = sorted(glob.glob(os.path.join(PREPROC_DIR, '*.npz')))
    subjects = [os.path.basename(f).split('.')[0] for f in files]
    
    if not files:
        print("No .npz files found! Run preprocessing first.")
        return

    print(f"Found {len(subjects)} subjects. Starting LOSO...")

    # 2. LOSO Loop
    for i, val_sid in enumerate(subjects):
        print(f"\n--- FOLD {i+1}/{len(subjects)}: Val={val_sid} ---")
        
        val_file = [f for f in files if val_sid in f]
        train_files = [f for f in files if val_sid not in f]
        
        train_ds = build_tf_dataset(train_files)
        val_ds = build_tf_dataset(val_file)
        
        # 3. Model
        model = build_lightweight_unet()
        model.compile(optimizer=tf.keras.optimizers.Adam(LR), 
                      loss=bce_dice_loss, metrics=[dice_coef])
        
        # 4. Callbacks
        ckpt_path = os.path.join(MODELS_DIR, f'best_{val_sid}.h5')
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_dice_coef', mode='max', save_best_only=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', patience=30, mode='max', restore_best_weights=True)
        ]
        
        # 5. Fit
        history = model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
                            validation_data=val_ds, validation_steps=VAL_PATCHES, callbacks=callbacks)
        
        # Save metrics
        best_score = max(history.history['val_dice_coef'])
        with open(os.path.join(MODELS_DIR, f'metrics_{val_sid}.json'), 'w') as f:
            json.dump({'subject': val_sid, 'dice': float(best_score)}, f)

if __name__ == "__main__":
    run_training()