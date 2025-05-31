import os
import numpy as np
from ridge_utils.ridge import bootstrap_ridge

input_dir='/ocean/projects/mth240012p/sapountz/lab3_fmri/results/ridge/inputs_4rdge/BERT'

X_train_z = np.load(os.path.join(input_dir, 'X_train_z.npy'), mmap_mode='r')
Y_train_z = np.load(os.path.join(input_dir, 'Y_train_z.npy'), mmap_mode='r')
X_test_z = np.load(os.path.join(input_dir, 'X_test_z.npy'), mmap_mode='r')
Y_test_z = np.load(os.path.join(input_dir, 'Y_test_z.npy'), mmap_mode='r')

# Print shapes of dataset
print("X_train_z shape:", X_train_z.shape)
print("Y_train_z shape:", Y_train_z.shape)
print("X_test_z shape:", X_test_z.shape)
print("Y_test_z shape:", Y_test_z.shape)


alphas = np.logspace(1, 4, 20)

wt, test_corrs, val_alphas, _, _ = bootstrap_ridge(
    X_train_z, Y_train_z[:, :45000],
    X_test_z, Y_test_z[:, :45000],
    alphas=alphas,
    nboots=20,
    chunklen=20,
    nchunks=1,
    normalpha=True,
    single_alpha=False
)



np.save("/ocean/projects/mth240012p/sapountz/lab3_fmri/results/ridge/ridge_boots/BERT/ridge_weights_45000-95000_date20250428.npy", wt)
np.save("/ocean/projects/mth240012p/sapountz/lab3_fmri/results/ridge/ridge_boots/BERT/ridge_test_corrs_45000-95000_date20250428.npy", test_corrs)
np.save("/ocean/projects/mth240012p/sapountz/lab3_fmri/results/ridge/ridge_boots/BERT/ridge_val_alphas_45000-95000_date20250428.npy", val_alphas)

print("wt shape:", wt.shape)
print("test_corrs shape:", test_corrs.shape)
print("val_alphas shape:", val_alphas.shape)

## Print Output Path ###
print("Saving outputs to:", os.getcwd())
