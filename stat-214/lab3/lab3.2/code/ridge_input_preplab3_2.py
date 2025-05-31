import os
import numpy as np

def load_and_standardize_inplace(story_shapes, x_mean, x_std, y_mean, y_std,
                                 subject_dir, embedding_dir):
    total_len = sum(n_time for (_, n_time, _, _) in story_shapes)
    feat_dim = story_shapes[0][2]
    voxel_dim = story_shapes[0][3]

    X_all = np.empty((total_len, feat_dim), dtype=np.float32)
    Y_all = np.empty((total_len, voxel_dim), dtype=np.float32)

    offset = 0
    for story, n_time, _, _ in story_shapes:
        x_path = os.path.join(embedding_dir, f"{story}.npy")
        y_path = os.path.join(subject_dir, f"{story}.npy")
        X = np.load(x_path, mmap_mode='r')[:n_time]
        Y = np.load(y_path, mmap_mode='r')[:n_time]

        X_z = (X - x_mean) / (x_std + 1e-8)
        Y_z = (Y - y_mean) / (y_std + 1e-8)

        X_all[offset:offset + n_time] = X_z
        Y_all[offset:offset + n_time] = Y_z
        print(f"Processed: {story} | n_time={n_time}")
        offset += n_time

    return X_all, Y_all

def compute_mean_std_manual(story_list, subject_dir, embedding_dir):
    n_total = 0
    x_sum = None
    x_sqsum = None
    y_sum = None
    y_sqsum = None
    valid_shapes = []

    for story in story_list:
        x_path = os.path.join(embedding_dir, f"{story}.npy")
        y_path = os.path.join(subject_dir, f"{story}.npy")
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            continue

        X = np.load(x_path, mmap_mode='r')
        Y = np.load(y_path, mmap_mode='r')
        n_time = min(X.shape[0], Y.shape[0])
        X = X[:n_time]
        Y = Y[:n_time]

        if x_sum is None:
            x_sum = np.zeros(X.shape[1], dtype=np.float64)
            x_sqsum = np.zeros(X.shape[1], dtype=np.float64)
            y_sum = np.zeros(Y.shape[1], dtype=np.float64)
            y_sqsum = np.zeros(Y.shape[1], dtype=np.float64)

        x_sum += X.sum(axis=0)
        x_sqsum += (X ** 2).sum(axis=0)
        y_sum += Y.sum(axis=0)
        y_sqsum += (Y ** 2).sum(axis=0)
        n_total += n_time
        valid_shapes.append((story, n_time, X.shape[1], Y.shape[1]))

    x_mean = x_sum / n_total
    x_std = np.sqrt(x_sqsum / n_total - x_mean**2)
    y_mean = y_sum / n_total
    y_std = np.sqrt(y_sqsum / n_total - y_mean**2)

    return x_mean, x_std, y_mean, y_std, valid_shapes


## Case: subject2, BERT

subject = "subject2"
embedding_type = "BERT"
subject_dir = "/ocean/projects/mth240012p/sapountz/lab3_fmri/data/subject2"
embedding_dir = "/ocean/projects/mth240012p/sapountz/lab3_fmri/results/embeddings/encoder"

story_files = [f[:-4] for f in os.listdir(subject_dir) if f.endswith(".npy")]
from sklearn.model_selection import train_test_split
train_stories, test_stories = train_test_split(story_files, test_size=0.2, random_state=42)

x_mean, x_std, y_mean, y_std, train_shapes = compute_mean_std_manual(train_stories, subject_dir, embedding_dir)
X_train_z, Y_train_z = load_and_standardize_inplace(train_shapes, x_mean, x_std, y_mean, y_std, subject_dir, embedding_dir)
_, _, _, _, test_shapes = compute_mean_std_manual(test_stories, subject_dir, embedding_dir)
X_test_z, Y_test_z = load_and_standardize_inplace(test_shapes, x_mean, x_std, y_mean, y_std, subject_dir, embedding_dir)

print("Train shape:", X_train_z.shape, Y_train_z.shape)
print("Test shape:", X_test_z.shape, Y_test_z.shape)

print("Data loading and standardization completed successfully for BERT.")

output_dir = "/ocean/projects/mth240012p/sapountz/lab3_fmri/results/ridge/inputs_4rdge/BERT"
os.makedirs(output_dir,exist_ok=True)

np.save(os.path.join(output_dir, "X_train_z.npy"), X_train_z)
np.save(os.path.join(output_dir, "Y_train_z.npy"), Y_train_z)
np.save(os.path.join(output_dir, "X_test_z.npy"), X_test_z)
np.save(os.path.join(output_dir, "Y_test_z.npy"), Y_test_z)

print("Saved standardized data to:", output_dir)


