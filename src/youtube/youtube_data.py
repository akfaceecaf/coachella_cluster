import pandas as pd
import yt_dlp
from pandas import Series


def search_youtube(search_qry : str, max_results : int=5):
    ydl_opts = dict(skip_download=True,
                    quiet=True,
                    limit=max_results,
                    default_search='ytmsearch',
                    noplaylist=True,
                    no_warnings = True)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        response = ydl.extract_info(f"ytsearch{max_results}:{search_qry}", download = False)
    if 'entries' in response:
        result = response['entries']
        return result
    else:
        return []

def extract_song_url(song : str, artist : str) -> Series:
    max_results = 1
    search_qry = f'{song} - {artist} (official audio)'

    try:
        results = search_youtube(search_qry, max_results)
        if results:
            result = results[0]
            print(f'Result found for:', search_qry)
            return pd.Series({
                'youtube_title': result['title'],
                'channel': result['channel'],
                'duration': result['duration'],
                'views': result['view_count'],
                'categories': result['categories'],
                'tags': result['tags'],
                'url': result['original_url']
            })
        else:
            print('No results returned.')
            return pd.Series()
    except Exception as e:
        print('Error fetching results.', e)
        return pd.Series()

def search_youtube_url(url : str) -> Series:
    ydl_opts = dict(
                    cookiesfrombrowser=('chrome',),
                    skip_download=True,
                    quiet=True)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url=url, download=False)
        if result:
            print(f'Result found for:', url)
            return pd.Series({
                'youtube_title': result['title'],
                'channel': result['channel'],
                'duration': result['duration'],
                'views': result['view_count'],
                'categories': result['categories'],
                'tags': result['tags'],
                'url': result['original_url']
            })
        else:
            print('No results returned.')
            return pd.Series()
    except Exception as e:
        print('Unable to extract data from url.')
        return pd.Series()