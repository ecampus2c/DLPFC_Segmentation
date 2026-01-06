import os
import glob
import numpy as np
import nibabel as nib
import tensorflow as tf
from src.config import *
from src.data_loader import make_patches, reconstruct_volume
from src.utils import bce_dice_loss, dice_coef

def run_inference():
    files = sorted(glob.glob(os.path.join(PREPROC_DIR, '*.npz')))
    custom_objs = {'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef}
    
    for f in files:
        sid = os.path.basename(f).split('.')[0]
        model_path = os.path.join(MODELS_DIR, f'best_{sid}.h5')
        
        if not os.path.exists(model_path):
            print(f"Skipping {sid}, no model found.")
            continue
            
        print(f"Predicting {sid}...")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)
        
        # Load Data
        d = np.load(f)
        img, aff = d['image'], d['affine']
        
        # Patching
        step = (PATCH_SIZE[0]//2, PATCH_SIZE[1]//2, PATCH_SIZE[2]//2)
        patches, grid = make_patches(img, PATCH_SIZE, step=step)
        patches = patches[..., np.newaxis]
        
        # Predict
        preds = model.predict(patches, batch_size=4, verbose=1)
        
        # Fix dimensions if needed (N, Z, Y, X, 1) -> (N, Z, Y, X)
        if preds.ndim == 5: preds = preds[..., 0]
        
        # Reconstruct
        recon = reconstruct_volume(preds, grid, img.shape)
        binary_mask = (recon > 0.5).astype(np.uint8)
        
        # Save
        nib.save(nib.Nifti1Image(binary_mask, aff), os.path.join(PREDS_DIR, f'pred_{sid}.nii.gz'))
        print(f"Saved pred_{sid}.nii.gz")

if __name__ == "__main__":
    run_inference()
