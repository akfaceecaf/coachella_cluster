import essentia as es
from essentia.standard import MonoLoader, TensorflowPredict2D
es.log.infoActive = False
es.log.warningActive = False
from src.utils import *
from src.youtube import *
from models import *
import os
import pandas as pd
import numpy as np
pd.options.display.max_columns = None
pd.set_option('display.expand_frame_repr', False)

MODELS_HOME = os.path.join(os.path.dirname(__file__), '..', 'models/')

class EmbeddingModel:
    def __init__(self, name : str, model_file : str, algorithm, **kwargs):
        self.MODELS_HOME = MODELS_HOME
        self.name = name
        self.model_file = model_file
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.model = self.load_model()

    def load_model(self):
        model = self.algorithm(graphFilename=os.path.join(self.MODELS_HOME, self.model_file), **self.kwargs)
        return model

    def get_embeddings(self, audio) -> np.array:
        # returns X-dimension feature vectors for every timeframe, more useful to feed audio data into model in this format rather than raw audio file
        embeddings = self.model(audio)
        return embeddings

class FeatureModel:
    def __init__(self, name : str, labels : list, model_file : str, algorithm, embedding_model = None, **kwargs):
        self.MODELS_HOME = MODELS_HOME
        self.name = name
        self.labels = labels
        self.model_file = model_file
        self.algorithm = algorithm
        self.embedding_model = embedding_model
        self.kwargs = kwargs
        self.model = self.load_model()

    def load_model(self):
        model = self.algorithm(graphFilename=os.path.join(self.MODELS_HOME, self.model_file), **self.kwargs)
        return model

    def get_predictions(self, audio) -> np.array:
        if self.embedding_model:
            embeddings = self.embedding_model.get_embeddings(audio)
            print('Using embedded audio format')
            return self.model(embeddings)
        print('Using audio file')
        return self.model(audio)

class FeatureExtractor:
    def __init__(self):
        self.embed_models = {}

    def add_embedding_model(self, name : str, model_file : str, algorithm, **kwargs):
        self.embed_models[name] = EmbeddingModel(name, model_file, algorithm, **kwargs)
        print(f'Successfully added model {name}')

    def extract_embedded_features(self, audio):
        embedding_vectors = []
        for n, em in self.embed_models.items():
            try:
                print(f'Extracting embeddings for {n}')
                embeddings = em.get_embeddings(audio)
                if embeddings is None:
                    raise ValueError(f'No values returned for model {n}.')
                embeddings = np.mean(embeddings, axis=0)
                embedding_vectors.extend(embeddings)
                print(f'Successfully added embedded features from {n}')
            except Exception as e:
                print(f'Failed to extract embeddings for audio file. {e}')
        return embedding_vectors

# DEV

# sample audio
# youtube_urls = load_data('youtube_urls_edited.csv')
# songs = load_data('songs_data_edited.csv', index_col = 0)
# tracks = youtube_urls.merge(songs, on='track_id', how='left')
# id = '7B3z0ySL9Rr0XvZEAjWZzM'
# sample = tracks[tracks['track_id']==id].iloc[0]

# yal = YoutubeAudioLoader()
# url = sample['url']
# yal.get_mp3_from_youtube(url=url,filename='temp')
# yal.convert_mp3_to_wav('temp.mp3', 'temp16000.wav', 16000)

# m1 = embed_models[0]
# fm1 = feature_models[0]
# em = EmbeddingModel(**m1)
# fm = FeatureModel(**fm1, embedding_model=em)

# audio processing, including ensuring audio is in single channel format
# sample_rate = 16000
# resample_quality = 4
# audio = MonoLoader(filename="temp16000.wav", sampleRate=sample_rate, resampleQuality=resample_quality)()
#
# extractor = FeatureExtractor()
# for model in embed_models:
#     extractor.add_embedding_model(**model)
#
# extractor.extract_embedded_features(audio)


# print(len(embedding_vectors))