import importlib.util
import subprocess
import sys

def install_if_missing(package):
    if importlib.util.find_spec(package) is None:
        print(f"Installing {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        print(f"{package} already installed.")

install_if_missing("gensim")
install_if_missing("joblib")
# install other necessary libraries

import os
import numpy as np
import pickle
import gensim.downloader as api
from preprocessing import downsample_word_vectors, make_delayed


# Load raw_text.pkl 
with open('../data/raw_text.pkl', 'rb') as f:
    raw_text = pickle.load(f)


# Load pre-trained Word2Vec model 
print("Loading Word2Vec model...")
w2v_model = api.load("word2vec-google-news-300")  # 300-dimensional
print("Word2Vec model loaded.")

# Build word-level vectors per story 
word2vec_vectors = {}
for story_name, story_data in raw_text.items():
    vectors = []
    for word in story_data.data:
        if word in w2v_model:
            vectors.append(w2v_model[word])
        else:
            vectors.append(np.zeros(300))  # OOV → zero vector
    word2vec_vectors[story_name] = np.vstack(vectors).astype(np.float32)

# Build dummy wordseqs for downsampling 
class DummyWordSeq:
    def __init__(self, data_times, tr_times):
        self.data_times = data_times
        self.tr_times = tr_times

wordseqs = {name: DummyWordSeq(story.data_times, story.tr_times)
            for name, story in raw_text.items()}

# Downsample word embeddings to TR times
word2vec_downsampled = downsample_word_vectors(
    stories=list(word2vec_vectors.keys()),
    word_vectors=word2vec_vectors,
    wordseqs=wordseqs
)

# Trim first 5 sec and last 10 sec
TR = np.mean(np.diff(next(iter(wordseqs.values())).tr_times))
n_skip_start = int(np.ceil(5 / TR))
n_skip_end = int(np.ceil(10 / TR))

for story in word2vec_downsampled:
    word2vec_downsampled[story] = word2vec_downsampled[story][n_skip_start:-n_skip_end]

# Add lag features
delays = [1, 2, 3, 4]
word2vec_delayed = {story: make_delayed(features, delays)
                    for story, features in word2vec_downsampled.items()}

# Save to file
os.makedirs('../results/embeddings/word2vec', exist_ok=True)
for story, X in word2vec_delayed.items():
    np.save(f'../results/embeddings/word2vec/{story}.npy', X)

print("Word2Vec embedding done. Saved to results/embeddings/word2vec/")


