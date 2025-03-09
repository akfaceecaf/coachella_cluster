import argparse
import os
import json
import pandas as pd
from src.spotify import SpotifyAuth, SpotifyData
from src.utils import save_data, load_data

DATA_PATH = os.path.join(os.path.dirname(__file__),'..','data/')
# Initialize Spotify Connection
auth = SpotifyAuth()
sp = SpotifyData(auth.access_token, auth.user_id)

overrides = {
    'GEL': ['GEL', '1fRv9jiRIN7zAOSpOfRP73']
}

# Override Artist ID
def override_artist_values(df: pd.DataFrame, artist_name: str, override_values: list):
    df.iloc[df['name'] == artist_name, 1:] = override_values

def get_spotify_artist_data():
    parser = argparse.ArgumentParser(description="Extract Spotify artist and top tracks data.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Input file of artist list."
    )
    parser.add_argument(
        "--overrides",
        action='store_true',
        help="Overrides for artist_id."
    )
    args = parser.parse_args()

    try:
        file_path = DATA_PATH + args.input_file
        with open(file_path, 'r') as f:
            artist_list = json.load(f)
    except Exception as e:
        print(f"Error opening file. {e}")
    print("Artist list successfully loaded.")

    # Get Spotify ID for every artist
    artists = sp.get_multiple_artists(artist_list)
    save_data(artists, 'artists_data.csv')

    if args.overrides:
        print("Performing artist overrides.")
        for artist, values in overrides.items():
            override_artist_values(artists, artist, values)
            save_data(artists, 'artists_data_edited.csv', index=False)
            print("Artist overrides completed.")

if __name__ == '__main__':
    get_spotify_artist_data()