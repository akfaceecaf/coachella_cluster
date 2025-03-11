import argparse
import os
import pandas as pd
from src.utils import load_data, save_data
from src.youtube import extract_song_url

def extract_youtube_urls():
    parser = argparse.ArgumentParser(description='get youtube urls for every song.')
    parser.add_argument('input_file',
                        type=str,
                        help='dataframe of songs')
    parser.add_argument('--overrides',
                        type=str,
                        help='override urls')
    args = parser.parse_args()

    songs = load_data(args.input_file, index_col = 0)
    songs['url'] = None

    if args.overrides:
        print('Performing url overrides...')
        url_overrides = load_data('url_overrides.csv')
        for i, row in url_overrides.iterrows():
            songs.loc[songs['track_id'] == row['track_id'], 'url'] = row['new_url']

    youtube_urls = songs.apply(lambda x: extract_song_url(x['track_name'],x['name'], x['url']), axis=1)
    failed_songs = youtube_urls[youtube_urls.isna().all(axis=1)==True].index
    success_songs = youtube_urls[youtube_urls.isna().all(axis=1)==False].index
    youtube_urls = pd.concat([songs['track_id'], youtube_urls], axis=1)

    print('Failed to fetch songs:', len(failed_songs))
    print('Successfully fetched songs:', len(success_songs))
    youtube_urls = pd.concat([songs['track_id'], youtube_urls], axis=1)
    save_data(youtube_urls, 'youtube_urls.csv', index=False)

if __name__ == '__main__':
    extract_youtube_urls()