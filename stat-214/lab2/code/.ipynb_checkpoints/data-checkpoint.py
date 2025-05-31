import glob
import numpy as np
import os
import pandas as pd

### NO label data process (161)############################

def make_data(patch_size=9):
    """
    Load the no-label image data and create patches from it.
    Args:
        patch_size: The size of the patches to create.
    Returns:
        images_long: A list of numpy arrays of the original images.
        patches: A list of lists of patches for each image.
    """

    # load images

    # filepaths = sorted(glob.glob("../data/image_data/*.npz"))

    # labeled_files = {"O013257.npz", "O013490.npz", "O012791.npz"}
    # filepaths = [fp for fp in filepaths if not any(lf in fp for lf in labeled_files)]

    filepaths = sorted(glob.glob("./data/*.npz"))

    ## this two lines are to filter out the 3 labeled image!
    ## when we want to load the full data, remove them
    
    # labeled_files = {"O013257.npz", "O013490.npz", "O012791.npz"}
    # filepaths = [fp for fp in filepaths if not any(lf in fp for lf in labeled_files)]

    
    print(f"files: {len(filepaths)} terms")
    images_long = []
    
    for fp in filepaths:
        npz_data = np.load(fp)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        if data.shape[1] == 11:
            data = data[:, :-1]  # remove labels
        images_long.append(data)

    # Compute global min and max for x and y over all images
    all_y = np.concatenate([img[:, 0] for img in images_long]).astype(int)
    all_x = np.concatenate([img[:, 1] for img in images_long]).astype(int)
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height = int(global_maxy - global_miny + 1)
    width = int(global_maxx - global_minx + 1)

    save_dir = "data/processed_images/"
    os.makedirs(save_dir, exist_ok=True)
    
    nchannels = images_long[0].shape[1] - 2
    print(f"Processing {len(images_long)} images...")
    
    for i, img in enumerate(images_long):
        if i % 10 == 0:
            print(f'Processing image {i}/{len(images_long)}')  
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)
        y_rel = y - global_miny
        x_rel = x - global_minx
        # avoid too much memory usage
        image = np.lib.format.open_memmap(f"{save_dir}/image_{i+1}.npy", mode='w+', dtype=np.float32, shape=(nchannels, height, width))
        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]
        for c in range(nchannels):
            image[c, y_valid, x_valid] = img_valid[:, c + 2]
        image.flush()  
        del image 
    
    print(' Finished reshaping images. Saved to disk.')

    image_files = sorted(glob.glob("data/processed_images/image_*.npy"))
    images = np.array([np.load(f, mmap_mode="r", allow_pickle=True) for f in image_files])

    
    # # Reshape each image onto the common grid.
    # nchannels = images_long[0].shape[1] - 2
    # images = []
    # for img in images_long:
    #     y = img[:, 0].astype(int)
    #     x = img[:, 1].astype(int)
    #     # Use global minimums to get relative coordinates.
    #     y_rel = y - global_miny
    #     x_rel = x - global_minx
    #     image = np.zeros((nchannels, height, width))
    #     valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
    #     y_valid = y_rel[valid_mask]
    #     x_valid = x_rel[valid_mask]
    #     img_valid = img[valid_mask]
    #     for c in range(nchannels):
    #         image[c, y_valid, x_valid] = img_valid[:, c + 2]
    #     images.append(image)
    # print('done reshaping images')

    # Now that all images have the same shape, convert to a 4D array.
    #images = np.array(images)
    
    pad_len = patch_size // 2

    # Global normalization across images.
    means = np.mean(images, axis=(0, 2, 3))[:, None, None]
    stds = np.std(images, axis=(0, 2, 3))[:, None, None]
    images = (images - means) / stds

    patches = []
    for i in range(len(images_long)):
        if i % 10 == 0:
            print(f'working on image {i}')
        patches_img = []
        # Pad the image by reflecting across the border.
        img_mirror = np.pad(
            images[i],
            ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
            mode="reflect",
        )
        # Use global min values to compute relative indices.
        ys = images_long[i][:, 0].astype(int)
        xs = images_long[i][:, 1].astype(int)
        for y, x in zip(ys, xs):
            y_idx = int(y - global_miny + pad_len)
            x_idx = int(x - global_minx + pad_len)
            patch = img_mirror[
                :,
                y_idx - pad_len : y_idx + pad_len + 1,
                x_idx - pad_len : x_idx + pad_len + 1,
            ]
            patches_img.append(patch.astype(np.float32))
        patches.append(patches_img)

    return images_long, patches



### Label data, fine-tuning ############################

def make_data_label(patch_size=9):
    """
    Load labeled training CSVs for image15/17/18.
    For each image, extract per-pixel 9x9 patches and keep only labeled pixels (label ≠ 0).
    
    Returns:
        patches: list of patches (shape (8, 9, 9))
        labels: list of labels (+1 or -1, float)
    """

    csv_files = sorted(glob.glob('../data_labeled_raw/image*_raw.csv'))
    print(f"Found {len(csv_files)} CSV training files.")
    
    df_all = []

    for fp in csv_files:
        df = pd.read_csv(fp)
        df = df[df['label'] != 0]  # only keep labeled pixels
        df_all.append(df)

    # Compute grid
    all_y = pd.concat(df_all)['y'].astype(int).to_numpy()
    all_x = pd.concat(df_all)['x'].astype(int).to_numpy()
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height = global_maxy - global_miny + 1
    width = global_maxx - global_minx + 1

    channels = ['NDAI', 'SD', 'CORR',
                'Radiance_angle_DF', 'Radiance_angle_CF',
                'Radiance_angle_BF', 'Radiance_angle_AF', 'Radiance_angle_AN']
    nchannels = len(channels)

    print(f"Normalizing {nchannels} channels over shape {height}x{width}...")

    full_image = np.zeros((nchannels, height, width))
    for df in df_all:
        for i, ch in enumerate(channels):
            y = df['y'].astype(int) - global_miny
            x = df['x'].astype(int) - global_minx
            full_image[i, y, x] = df[ch].to_numpy()

    means = np.mean(full_image, axis=(1, 2), keepdims=True)
    stds = np.std(full_image, axis=(1, 2), keepdims=True)
    full_image = (full_image - means) / stds

    pad_len = patch_size // 2
    padded_image = np.pad(full_image, ((0, 0), (pad_len, pad_len), (pad_len, pad_len)), mode='reflect')

    patches = []
    labels = []

    for i, df in enumerate(df_all):
        print(f"Processing image {i + 1}/{len(df_all)}...")

        ys = df['y'].astype(int).to_numpy()
        xs = df['x'].astype(int).to_numpy()
        ls = df['label'].to_numpy()

        for y, x, label in zip(ys, xs, ls):
            y_idx = y - global_miny + pad_len
            x_idx = x - global_minx + pad_len
            patch = padded_image[
                :,
                y_idx - pad_len : y_idx + pad_len + 1,
                x_idx - pad_len : x_idx + pad_len + 1
            ]
            patches.append(patch.astype(np.float32))
            labels.append(float(label)) 

    return patches, labels




