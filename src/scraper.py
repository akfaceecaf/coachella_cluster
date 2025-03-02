from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import os
import json

class CoachellaScraper:
    def __init__(self, url):
        self.url = url

        chrome_options = Options()
        chrome_options.add_argument("--headless")    
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def load_page(self):
        # load coachella artists page
        self.driver.get(self.url)
        time.sleep(5)

    def fetch_artist_list(self) -> list:
        artist_list = []
        # Get artistTile class
        artist_elements = self.driver.find_elements(By.CLASS_NAME, 'artistTile')
        
        # in artistTile class get title name of front side
        for artist in artist_elements:
            name = artist.find_elements(By.CSS_SELECTOR, 'div.card_face.card_face--front.cursor-pointer')
            name = list(set([n.text.strip() for n in name]))
            artist_list.extend(name)
        return artist_list

    def close_scraper(self):
        print('Closing scraper...')
        self.driver.quit()

    def save_artists(self, artist_list : list, overwrite : bool = False):
        filename = 'coachella_artists.json'
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..' , 'data/', filename)

        if not os.path.exists(filepath):
            print('Creating artist list file...')
            with open(filepath, 'w', encoding = 'utf-8') as f:
                json.dump(artist_list, f, indent = 4)
            print('Artist list data file successfully saved.')
        elif overwrite:
            print('Overwriting existing artist list file...')
            with open(filepath, 'w', encoding = 'utf-8') as f:
                json.dump(artist_list, f, indent = 4)
            print('Artist list data file successfully saved.')
        else:
            print('Artist list file already exists.')