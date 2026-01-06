import os
import random
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import tensorflow as tf
from patchify import patchify, unpatchify
from src.config import *

# --- IO & Preprocessing ---
def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata(dtype=np.float32), nii.affine

def sitk_resample_to_spacing(numpy_img, original_affine, target_spacing=TARGET_SPACING, is_label=False):
    img = sitk.GetImageFromArray(numpy_img)
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()[::-1]
    
    # Calculate new size
    target_spacing_sitk = tuple(target_spacing[::-1])
    new_size = [int(np.round(orig_size[i] * (orig_spacing[i] / target_spacing_sitk[i]))) for i in range(3)]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing_sitk)
    resampler.SetSize(new_size[::-1])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
        
    resampled = resampler.Execute(img)
    return sitk.GetArrayFromImage(resampled)

def zscore_normalize(img):
    mask = img != 0
    vals = img[mask]
    if vals.size == 0: return img
    return (img - vals.mean()) / (vals.std() + 1e-8)

def save_npz(sid, img, lbl, affine):
    out_path = os.path.join(PREPROC_DIR, f"{sid}.npz")
    np.savez_compressed(out_path, image=img.astype(np.float32), label=lbl.astype(np.uint8), affine=affine)
    return out_path

# --- Patching & Generator ---
def make_patches(volume, patch_size, step=None):
    if step is None: step = patch_size
    patches = patchify(volume, patch_size, step=step)
    nz, ny, nx, pz, py, px = patches.shape
    patches = patches.reshape(-1, pz, py, px)
    return patches, (nz, ny, nx)

def reconstruct_volume(patches, grid_shape, original_shape):
    nz, ny, nx = grid_shape
    pz, py, px = patches.shape[1:]
    patches_reshaped = patches.reshape(nz, ny, nx, pz, py, px)
    return unpatchify(patches_reshaped, original_shape)

def generator(npz_files, patches_per_subject):
    while True:
        random.shuffle(npz_files)
        for f in npz_files:
            d = np.load(f)
            img, lbl = d['image'], d['label']
            D, H, W = img.shape
            
            for _ in range(patches_per_subject):
                # Random crop logic with class balancing
                for _ in range(10): # try 10 times to find a positive patch
                    z = np.random.randint(0, max(1, D - PATCH_SIZE[0]))
                    y = np.random.randint(0, max(1, H - PATCH_SIZE[1]))
                    x = np.random.randint(0, max(1, W - PATCH_SIZE[2]))
                    
                    lb_p = lbl[z:z+PATCH_SIZE[0], y:y+PATCH_SIZE[1], x:x+PATCH_SIZE[2]]
                    
                    if random.random() > POS_NEG_RATIO or lb_p.sum() > 0:
                        im_p = img[z:z+PATCH_SIZE[0], y:y+PATCH_SIZE[1], x:x+PATCH_SIZE[2]]
                        
                        # Augmentation
                        if random.random() < 0.5: im_p, lb_p = im_p[::-1,:,:], lb_p[::-1,:,:]
                        if random.random() < 0.5: im_p, lb_p = im_p[:,::-1,:], lb_p[:,::-1,:]
                        
                        yield im_p[..., np.newaxis], (lb_p > 0).astype(np.float32)[..., np.newaxis]
                        break

def build_tf_dataset(npz_files):
    out_sig = (tf.TensorSpec((*PATCH_SIZE,1), tf.float32), tf.TensorSpec((*PATCH_SIZE,1), tf.float32))
    ds = tf.data.Dataset.from_generator(lambda: generator(npz_files, PATCHES_PER_SUBJECT), output_signature=out_sig)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)