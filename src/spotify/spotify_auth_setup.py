import os
from dotenv import load_dotenv, set_key, dotenv_values

class SpotifyAuthSetup:
    def __init__(self):
        self.env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config", "spotify.env")
        self.required_keys = ['CLIENT_ID','CLIENT_SECRET','REDIRECT_URI']

        # Check that an env file already exists, if not create a new one
        self.setup_env()

    def setup_env(self):
        if not os.path.exists(self.env_file):
            print("spotify.env file not found. Creating a new one...")
            open(self.env_file, 'w')
            print("spotify.env file created.")

        # Loading environment variables
        load_dotenv(self.env_file)
        env_variables = dotenv_values(self.env_file)

        missing_keys = [key for key in self.required_keys if not env_variables.get(key)]

        if missing_keys:
            print("Authentication setup required.")
            for key in self.required_keys:
                value = input(f"Enter {key}: ")
                set_key(self.env_file, key, value)
                os.environ[key] = value

            # Reload environment variables
            load_dotenv(self.env_file)

        else:
            print("Authentication setup has already been completed.")

    def update_credentials(self):
        if not os.path.exists(self.env_file):
            print("spotify.env file not found. Creating a new one...")
            open(self.env_file, 'w')
            print("spotify.env file created.")

        load_dotenv(self.env_file)

        print("Updating credentials:")
        for key in self.required_keys:
            value = input(f"Enter {key}: ")
            set_key(self.env_file, key, value)
            os.environ[key] = value

            # Reload environment variables
            load_dotenv(self.env_file)
        print("Credentials have been updated.")

    def get_credentials(self) -> dict:
        return {
            "CLIENT_ID" : os.getenv('CLIENT_ID'),
            "CLIENT_SECRET" : os.getenv('CLIENT_SECRET'),
            "REDIRECT_URI" : os.getenv('REDIRECT_URI')
        }