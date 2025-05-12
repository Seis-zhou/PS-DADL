import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Input, Dense, Permute, multiply, GlobalAveragePooling1D, Reshape, concatenate, Conv1D, Dropout, BatchNormalization, LeakyReLU, ReLU, ELU, GlobalMaxPooling1D, Flatten, AveragePooling1D, MaxPooling1D, Activation, PReLU, ELU, Add, Cropping1D, Multiply, Lambda, MultiHeadAttention, LayerNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler, Callback
from keras import optimizers, Model
from keras.models import load_model
from tensorflow.keras.utils import plot_model
import numpy as np
from numpy.random import seed
import math
from scipy import io
import scipy.io
import matplotlib.pyplot as plt
import h5py
import time
import hdf5storage
from Utils import *
from attention import *
from models import *
from math import log10
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import keras
import os
import gc
import seaborn as sns
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Dense, ReLU, concatenate
from tensorflow.keras.layers import Conv1D, ReLU, Dropout, concatenate
from tensorflow.keras.layers import Conv2D, Activation

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.run_functions_eagerly(True)
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

def noise_estimation_loss(noisy_input, denoised_output):
    residual_noise = noisy_input - denoised_output
    return tf.reduce_mean(tf.square(residual_noise))

class SNRCallback(Callback):
    def __init__(self, clean_data, noisy_data, batch_size, l1, l2, l3):
        super(SNRCallback, self).__init__()
        self.clean_data = clean_data
        self.noisy_data = noisy_data
        self.batch_size = batch_size
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.metrics = {'snr': [], 'psnr': [], 'ssim': [], 'mse': []}  # 用于存储每个 epoch 的指标
        self.best_snr = float('-inf')
        self.best_denoised_data = None

    def on_epoch_end(self, epoch, logs=None):
        predicted = self.model.predict(self.noisy_data, batch_size=self.batch_size)
        predicted_reshaped = np.transpose(predicted)
        denoised_data = yc_patch3d_inv(predicted_reshaped, n1, n2, n3, self.l1, self.l2, self.l3, s1, s2, s3)

        save_dir = './denoised_results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'denoised_data_epoch_{epoch + 1}.mat')
        io.savemat(save_path, {'data': denoised_data})

        snr_value = yc_snr(self.clean_data, denoised_data, 2)
        print(f"\nEpoch {epoch + 1}: Signal to Noise Ratio (SNR) = {snr_value:.4f} dB\n")
        self.metrics['snr'].append(snr_value)

        clean_data_reshape = np.reshape(self.clean_data, (n1, n2 * n3))
        denoised_data_reshape = np.reshape(denoised_data, (n1, n2 * n3))

        max_value = 255
        psnr_value_denoised = cal_psnr(denoised_data_reshape, clean_data_reshape, max_value)
        print(f"Epoch {epoch + 1}: Peak Signal-to-Noise Ratio (PSNR) of the denoised data = {psnr_value_denoised:.4f} dB\n")
        self.metrics['psnr'].append(psnr_value_denoised)
        ssim_value_denoised = cal_ssim(denoised_data_reshape, clean_data_reshape)
        print(f"Epoch {epoch + 1}: Structural Similarity Index (SSIM) of the denoised data = {ssim_value_denoised:.4f}\n")
        self.metrics['ssim'].append(ssim_value_denoised)
        mse_value_denoised = cal_mse(denoised_data_reshape, clean_data_reshape)
        print(f"Epoch {epoch + 1}: Mean Squared Error (MSE) of the denoised data = {mse_value_denoised:.6f}\n")
        self.metrics['mse'].append(mse_value_denoised)

        if snr_value > self.best_snr:
            self.best_snr = snr_value
            self.best_denoised_data = denoised_data.copy()

        print(f"Best SNR so far: {self.best_snr:.4f} dB")

        del predicted, predicted_reshaped, denoised_data, clean_data_reshape, denoised_data_reshape
        gc.collect()

def plot_curve(metric, label, ylabel, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(metric, label=label)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=600)
    plt.close()

output_dir = './Datasets/best_model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_name = 'original'
f = io.loadmat(f'./Datasets/{data_name}.mat')
Clean_data = f['data']
Clean_data = Clean_data[30:230, :48, :48]
Clean_data = yc_scale(Clean_data, 3)
print(f"Clean_data shape: {Clean_data.shape}")

n1, n2, n3 = np.shape(Clean_data)
io.savemat(f'./Datasets/{data_name}_clean.mat', mdict={'data': Clean_data}, appendmat=True)

Noisy_data = Clean_data + random_noise + err_n
io.savemat(f'./Datasets/{data_name}_noisy.mat', mdict={'data': Noisy_data}, appendmat=True)

snr_value1 = yc_snr(Clean_data, Noisy_data, 2)
print("Signal to Noise Ratio of the noisy data is {:.4f} dB".format(snr_value1))

A = np.array(Noisy_data)

param_sets = [(15, 1)]
for l1, s1 in param_sets:
    l2, l3, s2, s3 = l1, l1, s1, s1

    X = yc_patch3d(A, l1, l2, l3, s1, s2, s3)
    print(f"X shape: {X.shape}")
    X = np.transpose(X)
    file_path = './Datasets/Input_Patches_3D.h5'
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('data', data=X, compression="gzip")

    variances = np.var(X, axis=1)
    threshold = np.percentile(variances, 20)
    selected_blocks = variances > threshold
    X_new = X[selected_blocks, :]
    print(f"transpose X shape: {X.shape}")
    number = 50
    INPUT_SIZE1 = X_new.shape[0]
    INPUT_SIZE2 = X_new.shape[1]
    input_img = Input(shape=(INPUT_SIZE2,))

   


