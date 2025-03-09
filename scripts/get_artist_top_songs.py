import argparse
import os
import re
import json
import pandas as pd
from src.spotify import SpotifyAuth, SpotifyData
from src.utils import save_data, load_data

DATA_PATH = os.path.join(os.path.dirname(__file__),'..','data/')

# Initialize Spotify Connection
auth = SpotifyAuth()
sp = SpotifyData(auth.access_token, auth.user_id)

artists = load_data('artists_data_edited.csv', index_col=0)

## Remove multiple radio edits
def normalize_title(title: str):
    title = title.lower()
    title = re.split('-', title)[0].strip()
    return title

def extract_artist_top_tracks(artists : pd.DataFrame):
    parser = argparse.ArgumentParser(description="Extract top artist tracks from Spotify")
    parser.add_argument(
        "method",
        choices=['extract','preload'],
        help="Method of artist top songs extraction."
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Preloaded song list."
    )
    parser.add_argument(
        "--processing",
        action='store_true',
        help="perform additional processing"
    )
    args = parser.parse_args()

    if args.method == 'extract':
        # extract top tracks(up to 10) for each artist
        success_artists = []
        fail_artists = []
        songs_df = []
        for _, row in artists.iterrows():
            name = row['name']
            id = row['artist_id']
            top_tracks = sp.get_artist_top_tracks(id)
            if not top_tracks:
                fail_artists.append(name)
                print(f'Failed to get top songs for {name}.')
            else:
                for track in top_tracks:
                    track.update({'name': name})
                songs_df.extend(top_tracks)
                success_artists.append(name)
                print(f'Successfully uploaded top songs for {name}.')
        print(f'Failed artists:', len(fail_artists))
        print(f'Successful artists:', len(success_artists))
        songs = pd.DataFrame(songs_df)
        songs = songs.reindex(columns=['artist_id', 'name', 'track_id', 'track_name', 'track_artist', 'popularity', 'uri'])
        save_data(songs, 'songs_data.csv')
        songs = load_data('songs_data.csv', index_col=0)
    elif args.method == 'preload':
        if not args.file:
            parser.error("File not specified.")
        file_path = DATA_PATH + args.file
        songs = load_data(file_path, index_col=0)
    else:
        raise ValueError("Invalid method specified.")

    # Songs Adjustments
    songs = songs.sort_values(['name', 'track_name']).reset_index(drop=True)
    ## Remove duplicate songs
    songs = songs.drop_duplicates(subset='track_id')
    print(songs.shape)

    if args.processing:
        print("Performing additional processing...")
        songs['track_name_normalized'] = songs['track_name'].apply(normalize_title)
        songs = songs.drop_duplicates(subset=['artist_id', 'track_name'],keep='first').reset_index(drop=True)
        songs = songs.drop(columns='track_name_normalized')
        save_data(songs, 'songs_data_edited.csv')
        print("Processing completed.")

if __name__ == '__main__':
    extract_artist_top_tracks(artists)