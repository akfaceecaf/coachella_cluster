import os
import pandas as pd

def save_data(df : pd.DataFrame, filename : str, overwrite : bool = False, **kwargs):
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data/')
    filepath = dir + filename

    if os.path.exists(filepath):
        if overwrite: 
            print(f'Overwriting existing {filename}...')
            df.to_csv(filepath, **kwargs)
        else:
            raise Exception('File already exists and overwrite toggle is off.')
    else:
        print('Creating new...')
        df.to_csv(filepath, **kwargs)

def load_data(filename, **kwargs):
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data', filename)
    if os.path.exists(filepath):
        artists_df = pd.read_csv(filepath, **kwargs)
        return artists_df
    else:
        raise FileNotFoundError('Missing file.')

def create_unique_ids(df : pd.DataFrame) -> list[str]:
    return ['T' + str(i).zfill(4) for i in range(1, len(df) + 1)]