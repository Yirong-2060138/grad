## python3 embedding_GloVe.py 
## here we take the glove.6B.300d.txt as input


import os
import numpy as np
import pickle
from tqdm import tqdm
from preprocessing import downsample_word_vectors, make_delayed

# Load raw_text ===
with open('../data/raw_text.pkl', 'rb') as f:
    raw_text = pickle.load(f)


# Load GloVe embeddings ===
glove_path = '../data/glove.6B.300d.txt' 
print("Loading GloVe embeddings...")

glove_model = {}
with open(glove_path, 'r', encoding='utf8') as f:
    for line in tqdm(f):
        parts = line.strip().split()
        word = parts[0]
        vec = np.array(parts[1:], dtype=np.float32)
        glove_model[word] = vec

print(f"Loaded {len(glove_model)} GloVe vectors.")


# Build word-level vectors per story 
glove_vectors = {}
for story_name, story_data in raw_text.items():
    vectors = []
    for word in story_data.data:
        if word in glove_model:
            vectors.append(glove_model[word])
        else:
            vectors.append(np.zeros(300))  # OOV → zero vector
    glove_vectors[story_name] = np.vstack(vectors).astype(np.float32)

# Dummy wordseqs for downsampling 
class DummyWordSeq:
    def __init__(self, data_times, tr_times):
        self.data_times = data_times
        self.tr_times = tr_times

wordseqs = {name: DummyWordSeq(story.data_times, story.tr_times)
            for name, story in raw_text.items()}

# Downsample
glove_downsampled = downsample_word_vectors(
    stories=list(glove_vectors.keys()),
    word_vectors=glove_vectors,
    wordseqs=wordseqs
)

# Trim 5s start & 10s end
TR = np.mean(np.diff(next(iter(wordseqs.values())).tr_times))
n_skip_start = int(np.ceil(5 / TR))
n_skip_end = int(np.ceil(10 / TR))

for story in glove_downsampled:
    glove_downsampled[story] = glove_downsampled[story][n_skip_start:-n_skip_end]

# Add delays
delays = [1, 2, 3, 4]
glove_delayed = {story: make_delayed(features, delays)
                 for story, features in glove_downsampled.items()}

# Save to file
os.makedirs('../results/embeddings/glove', exist_ok=True)
for story, X in glove_delayed.items():
    np.save(f'../results/embeddings/glove/{story}.npy', X)

print("GloVe embedding done. Saved to results/embeddings/glove/")


