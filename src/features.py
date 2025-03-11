import essentia as es
from essentia.standard import MonoLoader, TensorflowPredict2D, RhythmExtractor2013
es.log.infoActive = False
es.log.warningActive = False
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

class ClassifierModel:
    def __init__(self, name : str, labels : list, model_file : str, algorithm, embedding_model=None, **kwargs):
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

    def get_predictions(self, audio) -> pd.DataFrame:
        if self.embedding_model:
            embeddings = self.embedding_model.get_embeddings(audio)
            print('Using embedded audio format')
            predictions = self.model(embeddings)
        else:
            print('Using audio file')
            predictions = self.model(audio)
        return pd.DataFrame(data=predictions, columns=self.labels)

class FeatureExtractor:
    def __init__(self):
        self.embed_models = {}
        self.classifier_models = {}

    def add_embedding_model(self, name : str, model_file : str, algorithm, **kwargs):
        self.embed_models[name] = EmbeddingModel(name, model_file, algorithm, **kwargs)
        print(f'Successfully added model {name}')

    def add_classifier_model(self, name : str, model_file : str, algorithm, labels : list, embedding_model=None, **kwargs):
        self.classifier_models[name] = ClassifierModel(name, labels, model_file, algorithm, embedding_model, **kwargs)
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

    def extract_classifier_features(self, audio) -> pd.Series:
        classifier_features = pd.Series()
        for n, cm in self.classifier_models.items():
            try:
                print(f'Extracting embeddings for {n}')
                predictions = cm.get_predictions(audio)
                if predictions.empty:
                    raise ValueError(f'No values returned for model {n}.')
                predictions = predictions.mean(axis=0)
                classifier_features = pd.concat([classifier_features, predictions])
                print(f'Successfully added embedded features from {n}')
            except Exception as e:
                print(f'Failed to extract embeddings for audio file. {e}')
        return classifier_features

    def extract_bpm(self, audio):
        rhythm_extractor = RhythmExtractor2013(method="multifeature")
        bpm = rhythm_extractor(audio)[0]
        return pd.Series({'bpm':bpm})