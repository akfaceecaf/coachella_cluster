import pandas as pd
import requests
from numpy.ma.core import arange
import urllib
import os

class SpotifyData:
    def __init__(self, access_token):
        self.access_token = access_token
        self.base_url = 'https://api.spotify.com/v1/'
        self.headers = {"Authorization": f"Bearer {self.access_token}"}
        self.user_id = 'YOUR_USERID'

    def search_for_artist(self, artist) -> (str, str):
        artist_qry = urllib.parse.quote(artist)
        target_url = self.base_url + 'search/'
        query = f"?q={artist_qry}&type=artist&limit=10"
        target_url = target_url + query
        response = requests.get(target_url, headers = self.headers)
        artist_items = response.json()['artists']['items']
        if artist_items:
            return {'name' : artist,
                    'spotify_name' : artist_items[0]['name'],
                    'artist_id' : artist_items[0]['id']}

    def get_multiple_artists(self, artist_list : list) -> pd.DataFrame:
        artists = []
        for artist in artist_list:
            artist_data = self.search_for_artist(artist)
            artists.append(artist_data)
        artists = pd.DataFrame(artists)
        return artists    

    def search_for_song(self, query):
        query = urllib.parse.quote(query)
        target_url = self.base_url + 'search/'
        query = f"?q={query}&type=track&limit=1"
        target_url = target_url + query
        response = requests.get(target_url, headers = self.headers)
        track_items = response.json()['tracks']['items']
        if track_items:
            return {
                'artist_name' : track_items[0]['artists'][0]['name'],
                'track_name' : track_items[0]['name'],
                'track_id' : track_items[0]['id'],
                'uri' : track_items[0]['uri']
            }

    def get_artist_top_tracks(self, artist_id) -> list:
        target_url = self.base_url + f'artists/{artist_id}/top-tracks?country=US'
        response = requests.get(target_url, headers=self.headers)
        if response.status_code != 200:
            print(f'Invalid input. {response.json()}')
            return None
        else:
            tracks = response.json()['tracks']
            return [
                {
                'artist_id' : artist_id,
                'track_id' : track['id'],
                'track_name' : track['name'],
                'popularity' : track['popularity'],
                'track_artist' : track['artists'][0]['name'],
                'uri' : track['uri']
                }
                for track in tracks
            ]
    
    def get_track_info(self, track_id) -> dict | None:
        target_url = self.base_url + f'tracks/{track_id}/'
        response = requests.get(target_url, headers = self.headers)
        if response.status_code != 200:
            print(f'Invalid input. {response.json()}')
            return None
        else:
            track_details = response.json()
            return {
                'id' : track_details['id'],
                'name': track_details['name'],
                'popularity' : track_details['popularity'],
                'artist' : track_details['artists'][0]['name'],
                'uri': track_details['uri'],
            }

    def create_playlist(self, playlist_name) -> str:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "name": playlist_name,
            "description": 'description',
            "public": 'public'
        }

        target_url = self.base_url + f"users/{self.user_id}/playlists"
        response = requests.post(target_url, json = payload, headers=headers)
        if response.status_code == 201:
            print(f"Playlist {playlist_name} has been created.")
            return response.json()["id"]
        else:
            raise RuntimeError(f'Failed to create playlist: {response.json()}')

    def add_songs_to_playlist(self, uri_ids : list, playlist_id : str):
        print(f"Adding songs to playlist id {playlist_id}...")

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        url = self.base_url + f'playlists/{playlist_id}/'
        for i in arange(0, len(uri_ids), 100):
            track_group = uri_ids[i : i + 100]
            track_str = ','.join(track_group)
            target_url = url + f'tracks?uris={track_str}'
            response = requests.post(target_url, headers = headers)
            if response.status_code == 201:
                print(track_str)
                print(f"Songs have been added to playlist id {playlist_id}.")
            else:
                raise RuntimeError(f"Failed to add songs to playlist: {response.json()}")
