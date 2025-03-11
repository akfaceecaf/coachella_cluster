import argparse
import pandas as pd
import re
import os
from essentia.standard import MonoLoader
from src.utils import load_data, save_data
from src.youtube import YoutubeAudioLoader
from src.features import FeatureExtractor
from models import embed_models, classifier_models

resample_quality = 4

def generate_temp_mp3(yal, url : str):
    ## Export youtube video, one with 16k SR and 44.1k SR
    yal.get_mp3_from_youtube(url=url, filename='temp')
    yal.convert_mp3_to_wav('temp.mp3', 'temp16000.wav', 16000)
    yal.convert_mp3_to_wav('temp.mp3', 'temp44100.wav', 44100)
    print('Completed generating files.')

def load_essentia_models():
    print("Loading Essentia model feature extractor...")
    extractor = FeatureExtractor()

    for em in embed_models:
        extractor.add_embedding_model(**em)

    for cm in classifier_models:
        if re.search('embeddings', cm['name'].lower()):
            if re.search('MusicCNN', cm['name']):
                if 'MSD MusicCNN' in extractor.embed_models:
                    extractor.add_classifier_model(**cm, embedding_model=extractor.embed_models['MSD MusicCNN'])
                else:
                    raise Exception('MSD MusicCNN model not loaded.')
            if re.search('VGGish', cm['name']):
                if 'VGGish' in extractor.embed_models:
                    extractor.add_classifier_model(**cm, embedding_model=extractor.embed_models['VGGish'])
                else:
                    raise Exception('VGGish model not loaded.')
    print("Audio models successfully loaded.")
    return extractor

def extract_audio_features():
    parser = argparse.ArgumentParser(description='Extract audio features')
    parser.add_argument('file',
                        type=str,
                        help='input file to get urls to extract')
    args = parser.parse_args()

    embedded_data = []
    classifier_data = pd.DataFrame()
    failed_tracks = []

    print("Loading youtube audio extractor...")
    yal = YoutubeAudioLoader()
    print("Youtube audio extractor loaded.")

    extractor = load_essentia_models()

    youtube_urls = load_data(args.file)
    youtube_urls = youtube_urls.head()

    for _, row in youtube_urls.iterrows():
        tid = row['track_id']
        url = row['url']
        try:
            # generate mp3 files
            generate_temp_mp3(yal, url)

            # Extract Embedded Features
            audio = MonoLoader(filename="temp16000.wav", sampleRate=16000, resampleQuality=resample_quality)()
            embedded_features = extractor.extract_embedded_features(audio)
            embedded_features = [tid, *embedded_features]

            # Extract Classifier Feature
            classifier_features = extractor.extract_classifier_features(audio)

            # Extract BPM
            audio = MonoLoader(filename="temp44100.wav", sampleRate=44100, resampleQuality=resample_quality)()
            bpm = extractor.extract_bpm(audio)

            # Join Features
            all_features = pd.concat([pd.Series({'track_id':tid}),classifier_features, bpm])
            all_features = pd.DataFrame(all_features).T

            # Removing files
            print('Audio features successfully extracted. Removing temp files...')
            for tmp in ['temp.mp3', 'temp16000.wav', 'temp44100.wav']:
                os.remove(tmp)
            embedded_data.append(embedded_features)
            classifier_data = pd.concat([classifier_data,all_features], axis=0)
        except Exception as e:
            print(f'Failed to load data for track {row['track_id']}. Removing temp files...')
            for tmp in ['temp.mp3','temp16000.wav','temp44100.wav']:
                if os.path.exists(tmp):
                    os.remove(tmp)

    save_data(embedded_data, 'embedded_features.csv', index=False)
    save_data(classifier_data, 'classifier_features.csv', index=False)

if __name__ == '__main__':
    extract_audio_features()