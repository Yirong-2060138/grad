
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import downsample_word_vectors, make_delayed

# Load raw text 
with open('../data/raw_text.pkl', 'rb') as f:
    raw_text = pickle.load(f)  # dict: story_name -> { "text": ..., "data_times": ..., "tr_times": ... }


sample_story = list(raw_text.values())[0]
print(type(sample_story))
print(dir(sample_story))

print(type(sample_story.data))
print(sample_story.data[:10]) 

# Generate BoW vectorizer across all stories 
all_texts = [" ".join(story_data.data) for story_data in raw_text.values()]
vectorizer = CountVectorizer()
vectorizer.fit(all_texts)

# Create word-level embeddings for each story 
bow_word_vectors = dict()
for story_name, story_data in raw_text.items():
    words = story_data.data 
    word_sentences = [" ".join([w]) for w in words] 
    X = vectorizer.transform(word_sentences).toarray()
    bow_word_vectors[story_name] = X.astype(np.float32)

# Downsample to TR rate using Lanczos interpolation 
# Create wordseq-like data with data_times and tr_times
class DummyWordSeq:
    def __init__(self, data_times, tr_times):
        self.data_times = data_times
        self.tr_times = tr_times

wordseqs = {name: DummyWordSeq(story_data.data_times, story_data.tr_times)
            for name, story_data in raw_text.items()}

bow_downsampled = downsample_word_vectors(
    stories=list(bow_word_vectors.keys()),
    word_vectors=bow_word_vectors,
    wordseqs=wordseqs
)

# Trim first 5 sec and last 10 sec 
TR = np.mean(np.diff(next(iter(wordseqs.values())).tr_times))
n_skip_start = int(np.ceil(5 / TR))
n_skip_end = int(np.ceil(10 / TR))

for story in bow_downsampled:
    bow_downsampled[story] = bow_downsampled[story][n_skip_start:-n_skip_end]

# Create delayed features with delays = [1, 2, 3, 4] 
delays = [1, 2, 3, 4]
bow_delayed = {story: make_delayed(features, delays)
               for story, features in bow_downsampled.items()}

# Save the result to file 
os.makedirs('../results/embeddings/bow', exist_ok=True)
for story, X in bow_delayed.items():
    np.save(f'../results/embeddings/bow/{story}.npy', X)

print("BoW embedding processing done. Saved in results/embeddings/bow/")




