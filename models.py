from essentia.standard import TensorflowPredictMusiCNN,TensorflowPredictVGGish

embed_models = [
    # A Music embedding extractor based on auto-tagging with the 50 most common tags of the Million Song Dataset, 16000 SR
    dict(name='MSD MusicCNN',
         model_file='msd-musicnn-1.pb',
         algorithm=TensorflowPredictMusiCNN,
         output="model/dense/BiasAdd"),
    # audio embedding extractor trained to predict tags from Youtube videos, 16000 SR
    dict(name='VGGish',
         model_file='audioset-vggish-3.pb',
         algorithm=TensorflowPredictVGGish,
         output="model/vggish/embeddings"),
]
