#!/usr/bin/env python
"""
This script projects target images (from a given folder) into the latent space
of a pretrained StyleGAN (e.g. WikiArt) model using a refined optimization
that employs a perceptual loss (LPIPS/VGG-based) and noise regularization.
For each image, it saves the target, the final projected image, the optimized
latent vector (in dlatents.npz), and (optionally) a video showing optimization progress.
 
Usage Example:
  python project_folder.py --network=WikiArt5.pkl --data=your_data_folder --outdir=out_projections --save-video true --seed 303
"""

import os, glob, pickle, argparse, imageio
import numpy as np, PIL.Image, tensorflow as tf, tqdm, math

import dnnlib
import dnnlib.tflib as tflib

# Helper: convert string to boolean.
def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

#----------------------------------------------------------------------------
# Projector class (refined latent projection with perceptual loss and noise regularization)
class Projector:
    def __init__(self):
        self.num_steps               = 3000
        self.dlatent_avg_samples     = 30000
        self.initial_learning_rate   = 0.01
        self.initial_noise_factor    = 0.025
        self.lr_rampdown_length      = 0.25
        self.lr_rampup_length        = 0.05
        self.noise_ramp_length       = 0.75
        self.regularize_noise_weight = 1e5
        self.verbose                 = True

        self._Gs                 = None
        self._minibatch_size     = None
        self._dlatent_avg        = None
        self._dlatent_std        = None
        self._noise_vars         = None
        self._noise_init_op      = None
        self._noise_normalize_op = None
        self._dlatents_var       = None
        self._dlatent_noise_in   = None
        self._dlatents_expr      = None
        self._images_float_expr  = None
        self._images_uint8_expr  = None
        self._target_images_var  = None
        self._lpips              = None
        self._dist               = None
        self._loss               = None
        self._opt                = None
        self._lrate_in           = None
        self._opt_step           = None
        self._cur_step           = None

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def set_network(self, Gs, dtype='float16'):
        if Gs is None:
            self._Gs = None
            return
        # Clone the network so that noise is not randomized.
        self._Gs = Gs.clone(randomize_noise=False, dtype=dtype, num_fp16_res=0, fused_modconv=True)

        self._info(f'Computing W midpoint and stddev using {self.dlatent_avg_samples} samples...')
        latent_samples = np.random.RandomState(123).randn(self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None)
        dlatent_samples = dlatent_samples.astype(np.float32)
        self._dlatent_avg = np.mean(dlatent_samples, axis=0, keepdims=True)  # [1, 1, C]
        self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg)**2) / self.dlatent_avg_samples) ** 0.5
        self._info(f'std = {self._dlatent_std:g}')

        self._info('Setting up noise inputs...')
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        while True:
            n = f'G_synthesis/noise{len(self._noise_vars)}'
            if not n in self._Gs.vars:
                break
            v = self._Gs.vars[n]
            self._noise_vars.append(v)
            noise_init_ops.append(tf.assign(v, tf.random_normal(tf.shape(v), dtype=tf.float32)))
            noise_mean = tf.reduce_mean(v)
            noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
            noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
        self._noise_init_op = tf.group(*noise_init_ops)
        self._noise_normalize_op = tf.group(*noise_normalize_ops)

        self._info('Building image output graph...')
        self._minibatch_size = 1
        self._dlatents_var = tf.Variable(tf.zeros([self._minibatch_size] + list(self._dlatent_avg.shape[1:])), name='dlatents_var')
        self._dlatent_noise_in = tf.placeholder(tf.float32, [], name='noise_in')
        dlatents_noise = tf.random.normal(shape=self._dlatents_var.shape) * self._dlatent_noise_in
        self._dlatents_expr = self._dlatents_var + dlatents_noise
        self._images_float_expr = tf.cast(self._Gs.components.synthesis.get_output_for(self._dlatents_expr), tf.float32)
        self._images_uint8_expr = tflib.convert_images_to_uint8(self._images_float_expr, nchw_to_nhwc=True)

        proc_images_expr = (self._images_float_expr + 1) * (255 / 2)
        sh = proc_images_expr.shape.as_list()
        if sh[2] > 256:
            factor = sh[2] // 256
            proc_images_expr = tf.reduce_mean(tf.reshape(proc_images_expr, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]), axis=[3,5])

        self._info('Building loss graph...')
        self._target_images_var = tf.Variable(tf.zeros(proc_images_expr.shape), name='target_images_var')
        if self._lpips is None:
            with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/vgg16_zhang_perceptual.pkl') as f:
                self._lpips = pickle.load(f)
        self._dist = self._lpips.get_output_for(proc_images_expr, self._target_images_var)
        self._loss = tf.reduce_sum(self._dist)

        self._info('Building noise regularization graph...')
        reg_loss = 0.0
        for v in self._noise_vars:
            sz = v.shape[2]
            while True:
                reg_loss += tf.reduce_mean(v * tf.roll(v, shift=1, axis=3))**2 + tf.reduce_mean(v * tf.roll(v, shift=1, axis=2))**2
                if sz <= 8:
                    break
                v = tf.reshape(v, [1, 1, sz//2, 2, sz//2, 2])
                v = tf.reduce_mean(v, axis=[3,5])
                sz = sz // 2
        self._loss += reg_loss * self.regularize_noise_weight

        self._info('Setting up optimizer...')
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')
        self._opt = tflib.Optimizer(learning_rate=self._lrate_in)
        self._opt.register_gradients(self._loss, [self._dlatents_var] + self._noise_vars)
        self._opt_step = self._opt.apply_updates()
        self._cur_step = 0

    def start(self, target_images):
        assert self._Gs is not None
        self._info('Preparing target images...')
        target_images = np.asarray(target_images, dtype='float32')
        target_images = (target_images + 1) * (255 / 2)
        sh = target_images.shape
        assert sh[0] == self._minibatch_size
        if sh[2] > self._target_images_var.shape[2]:
            factor = sh[2] // self._target_images_var.shape[2]
            target_images = np.reshape(target_images, [-1, sh[1], sh[2]//factor, factor, sh[3]//factor, factor]).mean((3,5))
        self._info('Initializing optimization state...')
        dlatents = np.tile(self._dlatent_avg, [self._minibatch_size, 1, 1])
        tflib.set_vars({self._target_images_var: target_images, self._dlatents_var: dlatents})
        tflib.run(self._noise_init_op)
        self._opt.reset_optimizer_state()
        self._cur_step = 0

    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return 0, 0
        t = self._cur_step / self.num_steps
        dlatent_noise = self._dlatent_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length)**2
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp
        feed_dict = {self._dlatent_noise_in: dlatent_noise, self._lrate_in: learning_rate}
        _, dist_value, loss_value = tflib.run([self._opt_step, self._dist, self._loss], feed_dict)
        tflib.run(self._noise_normalize_op)
        self._cur_step += 1
        return dist_value, loss_value

    @property
    def cur_step(self):
        return self._cur_step

    @property
    def dlatents(self):
        return tflib.run(self._dlatents_expr, {self._dlatent_noise_in: 0})

    @property
    def images_uint8(self):
        return tflib.run(self._images_uint8_expr, {self._dlatent_noise_in: 0})

#----------------------------------------------------------------------------
# Process a single target image.
def project_single_image(network_pkl, target_fname, outdir, save_video, seed):
    tflib.init_tf({'rnd.np_random_seed': seed})
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)
    # Load and preprocess target image.
    target_pil = PIL.Image.open(target_fname)
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
    target_pil = target_pil.convert('RGB')
    target_pil = target_pil.resize((Gs.output_shape[3], Gs.output_shape[2]), PIL.Image.ANTIALIAS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    target_float = target_uint8.astype(np.float32).transpose([2,0,1]) * (2/255) - 1

    proj = Projector()
    proj.set_network(Gs)
    proj.start([target_float])
    os.makedirs(outdir, exist_ok=True)
    target_pil.save(os.path.join(outdir, 'target.png'))
    writer = None
    if save_video:
        writer = imageio.get_writer(os.path.join(outdir, 'proj.mp4'), mode='I', fps=60, codec='libx264', bitrate='16M')
    with tqdm.trange(proj.num_steps) as t:
        for step in t:
            assert step == proj.cur_step
            if writer is not None:
                frame = np.concatenate([target_uint8, proj.images_uint8[0]], axis=1)
                writer.append_data(frame)
            dist, loss = proj.step()
            t.set_postfix(dist=f'{dist[0]:.4f}', loss=f'{loss:.2f}')
    PIL.Image.fromarray(proj.images_uint8[0], 'RGB').save(os.path.join(outdir, 'proj.png'))
    np.savez(os.path.join(outdir, 'dlatents.npz'), dlatents=proj.dlatents)
    if writer is not None:
        writer.close()

#----------------------------------------------------------------------------
# Main: process all images in a folder.
def main():
    parser = argparse.ArgumentParser(
        description='Project target images to latent space using refined perceptual loss optimization.'
    )
    parser.add_argument('--network', required=True, help='Pretrained network pickle filename')
    parser.add_argument('--data', required=True, help='Directory containing target images')
    parser.add_argument('--outdir', required=True, help='Directory to save outputs')
    parser.add_argument('--save-video', type=_str_to_bool, default=False, help='Save an optimization video')
    parser.add_argument('--seed', type=int, default=303, help='Random seed')
    args = parser.parse_args()

    image_paths = sorted(glob.glob(os.path.join(args.data, '*.jpg')) +
                          glob.glob(os.path.join(args.data, '*.png')))
    print(f'Found {len(image_paths)} images in {args.data}.')
    for img_path in image_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_subdir = os.path.join(args.outdir, base)
        os.makedirs(out_subdir, exist_ok=True)
        print(f'\nProjecting {img_path} to {out_subdir}...')
        project_single_image(args.network, img_path, out_subdir, args.save_video, args.seed)

if __name__ == '__main__':
    main()
