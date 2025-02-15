#!/usr/bin/env python
"""
This script:
  1. Iterates over images in a local folder (e.g., "data/").
  2. For each image, it checks if an optimized latent (W⁺) has been saved
     in a subfolder (e.g. "out_proj1/<basename>/dlatent.npz"). If found, it loads that latent;
     otherwise, it projects the image into the latent space (using a simple MSE loss) and saves it.
  3. Generates textural variations by perturbing only the later layers of the latent.
  4. Combines the input image and generated variations into a grid image.
  
Note: This script uses a simple MSE loss for projection. For improved fidelity, consider using perceptual loss.
"""

import os
import glob
import argparse
import numpy as np
import PIL.Image
import tensorflow as tf
import math
import pickle
import dnnlib
import dnnlib.tflib as tflib

#------------------------------------------------------------------
def create_variation_grid(input_img, variation_imgs):
    """Creates a grid with the input image on the left and variations in one row."""
    w, h = input_img.size
    n = len(variation_imgs) + 1
    grid_img = PIL.Image.new("RGB", (w * n, h))
    grid_img.paste(input_img, (0, 0))
    for i, var in enumerate(variation_imgs):
        grid_img.paste(var, ((i + 1) * w, 0))
    return grid_img

#------------------------------------------------------------------
def project_image(Gs, target_tensor, img_idx, num_steps=500, learning_rate=0.01):
    """
    Projects the target image into W⁺ space using a simple MSE loss.
    The latent is initialized by tiling the network’s average latent.
    """
    dlatent_avg = Gs.get_var('dlatent_avg')
    dlatent_avg = tf.reshape(dlatent_avg, [1, 512])
    resolution = Gs.output_shape[-1]
    num_layers = 2 * int(math.log2(resolution)) - 2
    dlatent_init = tf.tile(dlatent_avg, [1, num_layers])
    dlatent_init = tf.reshape(dlatent_init, [1, num_layers, 512])
    
    with tf.variable_scope(f"project_{img_idx}"):
        dlatent_var = tf.get_variable("dlatent_var", initializer=dlatent_init)
    
    raw_out = Gs.components.synthesis.get_output_for(dlatent_var, randomize_noise=False)
    generated = tf.transpose(raw_out, [0, 2, 3, 1])
    loss = tf.reduce_mean(tf.square(generated - target_tensor))
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, var_list=[dlatent_var])
    
    sess = tf.get_default_session()
    uninit_names = sess.run(tf.report_uninitialized_variables())
    if uninit_names.size > 0:
        uninit_vars = [v for v in tf.global_variables()
                       if v.name.split(":")[0].encode() in uninit_names]
        if uninit_vars:
            print("Initializing variables:")
            for v in uninit_vars:
                print("  ", v.name)
            sess.run(tf.variables_initializer(uninit_vars))
    
    best_loss = float('inf')
    best_dlatent = None
    for step in range(num_steps):
        _, loss_val, dlatent_val = sess.run([train_op, loss, dlatent_var])
        if loss_val < best_loss:
            best_loss = loss_val
            best_dlatent = dlatent_val.copy()
        if step % 100 == 0:
            print(f"Image {img_idx}: step {step}, loss {loss_val:.4f}")
    return best_dlatent

#------------------------------------------------------------------
def generate_variations_images(Gs, dlatent, num_variations, variation_strength,
                               outdir, base_name, vary_start=None, vary_end=None):
    """
    Generates textural variations by adding noise to selected layers of the latent.
    If vary_start/vary_end are not provided, the later half of the layers are perturbed.
    """
    variation_imgs = []
    num_layers = dlatent.shape[1]
    # Default: vary the latter half layers.
    if vary_start is None or vary_end is None:
        vary_start = num_layers // 2
        vary_end = num_layers
    for i in range(num_variations):
        delta = np.zeros_like(dlatent)
        # Only perturb layers from vary_start to vary_end.
        delta[:, vary_start:vary_end, :] = np.random.randn(1, vary_end - vary_start, dlatent.shape[2]) * variation_strength
        new_dlatent = dlatent + delta
        img = Gs.components.synthesis.run(
            new_dlatent, randomize_noise=True,
            output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
        variation_img = PIL.Image.fromarray(img[0], 'RGB')
        variation_imgs.append(variation_img)
        fname = os.path.join(outdir, f"{base_name}_variation_{i:03d}.png")
        variation_img.save(fname)
        print(f"Saved variation: {fname}")
    return variation_imgs

#------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Project images into latent space, generate textural variations, and create a grid."
    )
    parser.add_argument('--network', required=True,
                        help="Path to the pretrained model pickle (e.g. WikiArt5.pkl)")
    parser.add_argument('--data', default="data", help="Directory with input images")
    parser.add_argument('--outdir', default="out_proj1", help="Directory to save outputs")
    parser.add_argument('--num_steps', type=int, default=500, help="Projection optimization steps")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for projection")
    parser.add_argument('--num_variations', type=int, default=5, help="Number of variations per image")
    parser.add_argument('--variation_strength', type=float, default=0.15, help="Noise strength added to latent")
    parser.add_argument('--vary_start', type=int, default=None, help="First layer index to perturb")
    parser.add_argument('--vary_end', type=int, default=None, help="Last layer index (exclusive) to perturb")
    parser.add_argument('--latent_dir', default="out_proj1", help="Base directory where latents are saved as subfolders")
    args = parser.parse_args()

    # Gather target image paths.
    image_paths = []
    for ext in ["*.jpg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(args.data, ext)))
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images in {args.data}.")

    for idx, img_path in enumerate(image_paths):
        print(f"\nProcessing image {idx}: {img_path}")
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        latent_subdir = os.path.join(args.latent_dir, base_name)
        latent_fname = os.path.join(latent_subdir, "dlatents.npz")
        os.makedirs(latent_subdir, exist_ok=True)
        
        with tf.Graph().as_default():
            tflib.init_tf()  # New default session
            sess = tf.get_default_session()
            print(f"Loading network from {args.network} ...")
            with dnnlib.util.open_url(args.network) as f:
                _, _, Gs = pickle.load(f)
            
            _, _, H, W = Gs.output_shape
            input_img = PIL.Image.open(img_path).convert("RGB").resize((W, H), PIL.Image.LANCZOS)
            target = np.array(input_img).astype(np.float32)
            target = target / 127.5 - 1.0
            target = target.reshape([1, H, W, 3])
            target_tensor = tf.constant(target, dtype=tf.float32)
            
            if os.path.exists(latent_fname):
                print(f"Loading saved latent from {latent_fname}")
                latent_data = np.load(latent_fname)
                best_dlatent = latent_data['dlatents']
            else:
                best_dlatent = project_image(Gs, target_tensor, idx,
                                             num_steps=args.num_steps,
                                             learning_rate=args.learning_rate)
                np.savez(latent_fname, dlatents=best_dlatent)
                print(f"Saved latent vector to {latent_fname}")
            
            # Generate variations using only the later layers (by default, later half).
            variation_imgs = generate_variations_images(Gs, best_dlatent, args.num_variations,
                                                         args.variation_strength,
                                                         latent_subdir, base_name,
                                                         vary_start=args.vary_start, vary_end=args.vary_end)
            
            grid_img = create_variation_grid(input_img, variation_imgs)
            grid_fname = os.path.join(latent_subdir, f"{base_name}_variation_grid.png")
            grid_img.save(grid_fname)
            print(f"Saved variation grid: {grid_fname}")
        tf.reset_default_graph()

if __name__ == "__main__":
    main()
