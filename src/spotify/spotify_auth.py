import os
from .spotify_auth_setup import SpotifyAuthSetup
import webbrowser
import requests
from dotenv import set_key

class SpotifyAuth:
    def __init__(self):
        # Load Spotify Authentication Credentials
        auth_setup = SpotifyAuthSetup()
        credentials = auth_setup.get_credentials()
        self.client_id = credentials['CLIENT_ID']
        self.client_secret = credentials['CLIENT_SECRET']
        self.redirect_uri = credentials['REDIRECT_URI']
        self.scope = "playlist-modify-public"
        self.env_file = auth_setup.env_file
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.refresh_token = os.getenv("REFRESH_TOKEN")

        if not self.access_token:
            # Get authorization code
            auth_code = self.get_authorization_code()
            self.get_access_token(auth_code)
        else:
            # Test access token is not expired / works
            self.test_access_token()

    def get_authorization_code(self):
        auth_url = f"https://accounts.spotify.com/authorize?client_id={self.client_id}&response_type=code&redirect_uri={self.redirect_uri}&scope={self.scope}"
        print("A link will open and you will be prompted to authorize the app. Paste the authorization code from the URL.")
        webbrowser.open(auth_url, new = 1)

        auth_code = input("Input authentication code: ").strip()
        print("Authentication code created.")
        return auth_code

    def get_access_token(self, auth_code):
        token_url = "https://accounts.spotify.com/api/token"
        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope" : self.scope
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(token_url, data=data, headers=headers)
        token_info = response.json()
        if 'access_token' in token_info:
            self.access_token = token_info['access_token']
            self.refresh_token = token_info['refresh_token']
            os.environ['ACCESS_TOKEN'] = self.access_token
            set_key(self.env_file, "ACCESS_TOKEN",self.access_token)
            os.environ['REFRESH_TOKEN'] = self.refresh_token
            set_key(self.env_file, "REFRESH_TOKEN", self.refresh_token)
            print("Tokens successfully retrieved.")
        else:
            raise RuntimeError("Failed to retrieve access token.")

    def refresh_access_token(self):
        token_url = "https://accounts.spotify.com/api/token"
        data = {
            "grant_type" : "refresh_token",
            "refresh_token" : self.refresh_token,
            "client_id" : self.client_id,
            "client_secret" : self.client_secret,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(token_url, data = data, headers = headers)
        token_info = response.json()
        if 'access_token' in token_info:
            self.access_token = token_info['access_token']
            set_key(self.env_file, "ACCESS_TOKEN", self.access_token)
            os.environ['ACCESS_TOKEN'] = self.access_token
            print("Tokens successfully refreshed.")
        else:
            raise RuntimeError(f"Failed to refresh token. {token_info}")

    def test_access_token(self):
        url = "https://api.spotify.com/v1/me"  # A simple request to check token validity
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.get(url, headers=headers)

        if response.status_code == 401:
            print("Access token expired. Refreshing...")
            return self.refresh_access_token()
        elif response.status_code == 200:
            print("Success")
            return self.access_token
        else:
            print(f"Error testing access token.{response.json()}")