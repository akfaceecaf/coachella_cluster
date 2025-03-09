import argparse
from scripts.run_scraper import run_scraper
# import matplotlib.pyplot as plt
# from PIL.Image import preinit
# from scipy.cluster.vq import kmeans
# from sklearn.metrics import silhouette_score
# from tensorflow.python.ops.distributions.util import embed_check_categorical_event_shape
#
# from src.scraper import *
# from src.spotify import *
# from src.utils import *
# from src.youtube import *
# from src.feature_extraction import MusicFeatureExtractor
# from src.features import FeatureExtractor
# import essentia as es
# es.log.infoActive = False
# es.log.warningActive = False
# from essentia.standard import TensorflowPredictMusiCNN, MonoLoader, TensorflowPredict2D
# from models import embed_models
# import pandas as pd
# import numpy as np
# pd.options.display.max_rows = None
# pd.options.display.max_columns = None
# pd.set_option('display.expand_frame_repr', False)
# import re


def main():
    # get youtube urls of songs
    # songs = load_data('songs_data_edited.csv', index_col = 0)
    # youtube_urls = songs.apply(lambda x: extract_song_url(x['track_name'],x['name']), axis=1)
    # failed_songs = youtube_urls[youtube_urls.isna().all(axis=1)==True].index
    # success_songs = youtube_urls[youtube_urls.isna().all(axis=1)==False].index
    # youtube_urls = pd.concat([songs['track_id'], youtube_urls], axis=1)
    # print('Failed to fetch songs:', len(failed_songs))
    # print('Successfully fetched songs:', len(success_songs))
    # youtube_urls = pd.concat([songs['track_id'], youtube_urls], axis=1)
    # save_data(youtube_urls, 'youtube_urls.csv', index = False)

    # # youtube url overrides
    # youtube_urls = load_data('youtube_urls.csv')
    #
    # # create override function
    # url_overrides_df = pd.read_csv('data/url_overrides.csv')
    # url_overrides = dict(zip(url_overrides_df['track_id'],url_overrides_df['new_url']))
    #
    # for track_id, new_url in url_overrides.items():
    #     result = search_youtube_url(new_url)
    #     if not result.empty:
    #         youtube_urls.loc[youtube_urls['track_id'] == track_id,
    #         ['youtube_title', 'channel', 'duration', 'views','categories', 'tags', 'url']] = pd.DataFrame([result], index=youtube_urls.loc[youtube_urls['track_id'] == track_id].index)
    #     else:
    #         youtube_urls.loc[youtube_urls['track_id'] == track_id,
    #         ['youtube_title', 'channel', 'duration', 'views', 'categories', 'tags', 'url']] = None
    #
    # save data as edited
    # save_data(youtube_urls, 'youtube_urls_edited.csv', overwrite=True,index=False)

    # Extract Audio Track Features
    # youtube_urls = load_data('youtube_urls_edited.csv')
    #
    # yal = YoutubeAudioLoader()
    #
    # extractor = FeatureExtractor()
    # for model in embed_models:
    #     extractor.add_embedding_model(**model)
    #
    # success_tracks = []
    # failed_tracks = []
    #
    # sample_rate = 16000
    # resample_quality = 4
    #
    # for _, row in youtube_urls.iterrows():
    #     tid = row['track_id']
    #     url = row['url']
    #     try:
    #         ## Export youtube video
    #         yal.get_mp3_from_youtube(url=url,filename='temp')
    #         yal.convert_mp3_to_wav('temp.mp3', 'temp16000.wav', 16000)
    #         print('Completed generating files.')
    #
    #         # Extract Essentia Model Features
    #         audio = MonoLoader(filename="temp16000.wav", sampleRate=sample_rate, resampleQuality=resample_quality)()
    #         embedded_features = extractor.extract_embedded_features(audio)
    #         embedded_features = [tid, *embedded_features]
    #         # Removing files
    #         print('Audio features successfully extracted. Removing temp files...')
    #         for tmp in ['temp.mp3', 'temp16000.wav']:
    #             os.remove(tmp)
    #         success_tracks.append(embedded_features)
    #     except Exception as e:
    #         print(f'Failed to load data for track {row['track_id']}. Removing temp files...')
    #         for tmp in ['temp.mp3','temp16000.wav']:
    #             if os.path.exists(tmp):
    #                 os.remove(tmp)
    #         failed_tracks.append(tid)
    # print('Successful tracks:', len(success_tracks))
    # print('Failed tracks:', len(failed_tracks))
    # print(success_tracks)
    #
    # success_tracks = pd.DataFrame(success_tracks)
    # save_data(success_tracks, 'embedded_features.csv', overwrite=True, index=False)

    # embedded_features = load_data('embedded_features.csv', index_col=0)
    # # returns 328D embedded feature vectors
    # # print(embedded_features.shape)
    # embedded_features.index.name = 'track_id'
    #
    # embedded_features = songs[['track_id','artist_id']].merge(embedded_features, how='outer',on='track_id',indicator=True)
    # embedded_features = embedded_features.groupby('artist_id').mean(numeric_only=True)
    # embedded_features.columns = [f'Feature {i}' for i in embedded_features.columns]
    # # print(embedded_features.head())
    # # print(embedded_features.shape)
    #
    # # Cluster Analysis
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.decomposition import PCA
    #
    # features = embedded_features.columns
    # X = embedded_features.values
    #
    # # scale data
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    #
    # components_range = range(1,min(len(features)+1,100))
    # variances = []
    # for c in components_range:
    #     pca = PCA(n_components=c)
    #     # reduce dimensionality into n components which are weighted sums of the original feature vectors
    #     pca.fit(X_scaled)
    #     variance = pca.explained_variance_ratio_.sum()
    #     variances.append(variance)
    #     print(f'Explained variance for {c} components: {variance:,.2%}')

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(14,10))
    # sns.lineplot(x=components_range, y=variances, marker='o', color='r')
    # plt.title('Explained Variance Ratio by Number of Components Using PCA')
    # xticks = [x for x in range(0,101,10)]
    # xticks[0] = 1
    # plt.xticks(xticks)
    # plt.xlabel('No. Components')
    # plt.ylabel('Explained Variance Ratio')
    # plt.show()

    # n_components = 20
    # pca = PCA(n_components=n_components)
    # X_pca = pca.fit_transform(X_scaled)
    # print(X_pca.shape)

    # Analyze What Features Have the Most Weight to Each Column
    # component = pca.components_[0]
    # idx = np.argsort(np.abs(component))[::-1][:10]
    # top_features = features[idx].values
    # weights = component[idx]
    # print('Features:', *top_features)
    # print('Weights:', *weights)

    # from sklearn.cluster import KMeans
    # from sklearn.metrics import silhouette_score
    # cluster_range = range(2, 20)
    # inertias = []
    # silhouette_scores = []
    # for c in cluster_range:
    #     kmeans = KMeans(n_clusters=c, random_state=42)
    #     labels = kmeans.fit_predict(X_pca)
    #     inertia = kmeans.inertia_
    #     inertias.append(inertia)
    #     s_score = silhouette_score(X_pca, labels)
    #     silhouette_scores.append(s_score)
    #     print(f'Inertia for {c} clusters: {inertia:,.0f}')
    #     print(f'Silhouette Score for {c} clusters: {s_score:,.2f}')

    # plt.figure(figsize=(14,10))
    # sns.lineplot(x=cluster_range, y=inertias, marker='o', color='r')
    # plt.title('Inertia by Number of Kmeans Clusters')
    # plt.xlabel('No. Clusters')
    # plt.xticks(cluster_range)
    # plt.ylabel('Inertia')
    # plt.show()

    # plt.figure(figsize=(14,10))
    # sns.lineplot(x=cluster_range, y=silhouette_scores, marker='o', color='r')
    # plt.title('Silhouette Score by Number of Kmeans Clusters')
    # plt.xlabel('No. Clusters')
    # plt.xticks(cluster_range)
    # plt.ylabel('Silhouette Score')
    # plt.show()

    # n_clusters = 4
    # kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    # labels = kmeans.fit_predict(X_pca)
    # labels = np.array([i+1 for i in labels])
    # inertia = kmeans.inertia_
    # s_score = silhouette_score(X_pca, labels)
    #
    # # # save model
    # # import pickle
    # # with open('data/kmeans_cluster_model1.sav', 'wb') as myFile:
    # #     pickle.dump(kmeans, myFile)
    # #
    # import plotly.express as px
    # n_components = 2
    # pca = PCA(n_components=n_components)
    # X_pca = pca.fit_transform(X_scaled)
    # plot_df = pd.DataFrame(index=embedded_features.index, data=X_pca, columns=['PCA1','PCA2'])
    # plot_df = plot_df.merge(songs[['artist_id','name']].drop_duplicates(), on='artist_id', how='outer')
    #
    # # simplified projection of seeing data using 2 PCA components (explains ~50% of variance)
    # fig = px.scatter(plot_df, x='PCA1',y='PCA2',
    #                  color=[str(x) for x in labels],
    #                  hover_data=['name'],
    #                  title='K-means Cluster Visualization (PCA 2D Projection)',
    #                  labels = {"color": "Cluster Group"})
    # fig.update_traces(marker=dict(size=8))
    # fig.show()

    # Create Playlists
    # temp = pd.concat([plot_df,pd.Series(labels,name='cluster')], axis=1)
    # temp2 = songs.merge(temp[['artist_id','cluster']].drop_duplicates(), on='artist_id',how='outer')
    # temp2.groupby(['cluster','name'])[['popularity']].mean().to_clipboard()
    # clusters = sorted(set(labels))
    # for c in clusters:
    #     cut = temp2[temp2['cluster']==c]
    #     uris = cut['uri'].to_list()
    #     playlist_id = sp.create_playlist(str(c))
    #     sp.add_songs_to_playlist(uris, playlist_id)

if __name__ == '__main__':
    main()
