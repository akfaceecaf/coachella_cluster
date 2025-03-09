import os
import json
import argparse
from src.utils import save_data, load_data, override_artist_list
from src.configs import BASE_URL
from scripts.run_scraper import run_scraper

DATA_PATH = os.path.join(os.path.dirname(__file__),'..','data/')

# Adjustments for multiple artists acts
exclusions = ["DIXON X JIMI JULES",
              "GUSTAVO DUDAMEL & LA PHIL",
              "MIND AGAINST X MASSANO",
              "PETE TONG X AHMED SPINS",
              "SEUN KUTI & EGYPT 80"]

inclusions = ['DIXON',
              'JIMI JULES',
              'GUSTAVO DUDAMEL',
              'LA PHIL',
              'MIND AGAINST',
              'MASSANO',
              'PETE TONG',
              'AHMED SPINS',
              'SEUN KUTI',
              'EGYPT 80']

def process_artists():
    parser = argparse.ArgumentParser(description="Process artist list, either by running Coachella page scraper, or pre-loaded list.")

    parser.add_argument(
        "method",
        choices=['scrape','preload'],
        help="Choose to scrape Coachella artist page or use pre-loaded list."
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Artist json file to read from (required if using preload argument)."
    )
    parser.add_argument(
        "--overrides",
        action='store_true',
        help="Override artist list."
    )

    args = parser.parse_args()
    file_path = DATA_PATH + "coachella_artists.json"

    if args.method == 'scrape':
        print("Running scraper...")
        run_scraper(BASE_URL)
        file_path = DATA_PATH + "coachella_artists.json"
        with open(file_path, 'r') as f:
            artist_list = json.load(f)
        print("File successfully loaded.")
    elif args.method == 'preload':
        if not args.file:
            parser.error("Pre-loaded file not specified.")
        print(f"Loading {args.file}...")
        file_name = args.file
        file_path = DATA_PATH + file_name
        with open(file_path, 'r') as f:
            artist_list = json.load(f)
        print("File successfully loaded.")
    else:
        raise ValueError("Invalid method specified.")
    if args.overrides:
        print("Performing artist overrides.")
        artist_list_updated = override_artist_list(artist_list, exclusions, inclusions)
        print("Overrides completed.")
        file_name = "coachella_artists_edited.json"
        file_path = DATA_PATH + file_name
        if not os.path.exists(file_path):
            print("Saving updated file...")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(artist_list_updated, f, indent=4)
            print("Saved.")
        else:
            print("File already exists.")

if __name__ == '__main__':
    process_artists()
