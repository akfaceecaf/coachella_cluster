import essentia.standard as es
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN
import numpy as np
import os

class Model:
    # model to have models pre-loaded
    def __init__(self, name : str, labels : list[str], model_file : str):
        self.MODELS_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..' , 'models/')
        self.name = name
        self.labels = labels
        self.model_file = model_file
        self.model = self.load_model()

    def load_model(self) -> object:
        model = TensorflowPredictMusiCNN(graphFilename=os.path.join(self.MODELS_HOME, self.model_file))
        return model

    def predict(self, input_file : str, sr : int):
        try:
            audio = MonoLoader(filename = input_file, sampleRate= sr, resampleQuality=4)()
            results = self.model(audio)
            average_predictions = np.mean(results, axis = 0)
            return dict(zip(self.labels, average_predictions))
        except Exception as e:
            print(f'Error fetching high_level features: {e}')
            return {}

class MusicFeatureExtractor:
    # feed multiple models and extract features
    def __init__(self):
        self.models = {}
        self.extractor = None

    def load_extractor(self):
        if not self.extractor:
            self.extractor = es.MusicExtractor()

    def add_model(self, name : str, labels : list[str], model_file : str):
        self.models[name] = Model(name, labels, model_file)
    
    def get_low_level_features(self, input_file):
        self.load_extractor()
        try:
            features, _ = self.extractor(input_file)
            return {
                'spectral_centroid_mean' : features['lowlevel.spectral_centroid.mean'],
                'mfcc_1': features['lowlevel.mfcc.mean'][0],
                'mfcc_2': features['lowlevel.mfcc.mean'][1],
                'mfcc_3': features['lowlevel.mfcc.mean'][2],
                'bpm' : features['rhythm.bpm']
            }
        except Exception as e:
            print(f'Error fetching low_level features: {e}')
            return {}

    def get_high_level_features(self, input_file, sr) -> dict:
        features = {}
        for model in self.models.values():
            features.update(model.predict(input_file, sr))
        return features


    # # temp solution for valence, arousal
    # embedding_model = TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb",
    #                                            output="model/dense/BiasAdd",
    #                                            patchHopSize=187)
    # valence_arousal_model = TensorflowPredict2D(graphFilename='models/emomusic-msd-musicnn-2.pb',
    #                                             output="model/Identity")
    # def get_valence_arousal(input_file : str, sr : int):
    #     labels = ["valence", "arousal"]
    #     try:
    #         audio = MonoLoader(filename=input_file, sampleRate=sr)()
    #         embeddings = embedding_model(audio)
    #         results = valence_arousal_model(embeddings)
    #         average_predictions = np.mean(results, axis = 0)
    #
    #         # Manual normalization (1, 9) -> (-1, 1)
    #         average_predictions = (average_predictions - 5) / 4
    #         return dict(zip(labels, average_predictions))
    #     except Exception as e:
    #         print(f'Error fetching high_level features: {e}')
    #         return {}