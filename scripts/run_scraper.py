from src.scraper import CoachellaScraper
from src.configs import BASE_URL

def run_scraper(url : str):
    # Initialize scraper and get artist list data
    scraper = CoachellaScraper(url)
    scraper.load_page()
    print("Fetching artist list...")
    artist_list = scraper.fetch_artist_list()
    print("Artist list extracted. Closing scraper...")
    scraper.close_scraper()
    scraper.save_artists(artist_list)

if __name__ == '__main__':
    run_scraper(BASE_URL)