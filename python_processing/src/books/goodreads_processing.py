import pandas as pd
import numpy as np
import subprocess
import requests
import time
from utils import clean_rename_move_file

fiction_genres = ['drama', 'horror', 'thriller']

def add_dates_read():
    """Load Goodreads export and input Excel files"""
    df = pd.read_csv("files/exports/goodreads_exports/gr_export.csv")
    df2 = pd.read_excel('files/work_files/gr_work_files/gr_dates_input.xlsx')
    df2["Book Id"].fillna(0, inplace=True)
    d = pd.merge(df, df2, on = "Book Id", how = 'left')
    d = d[(d['Exclusive Shelf'] == 'read') & (d['Check'].isna())].reset_index(drop = True)
    # Prompt user to input reading dates for each book
    if d.shape[0] == 0:
        return df, df2
    print(f"There are {d.shape[0]} book(s) where reading dates must be added")
    subprocess.run(['open', '-a', 'Firefox', '-g', "https://www.goodreads.com/review/list/143865509?ref=nav_mybooks"])
    for index, row in d.iterrows():
        row_num = df2.shape[0] + 1 + index
        df2.loc[row_num, 'Book Id'] = row['Book Id']
        df2.loc[row_num, 'Title'] = row['Title_x']
        df2.loc[row_num, 'Date started'] = pd.to_datetime(input(f"When did you start {row['Title_x']} ? dd/mm/YYYY "), format='%d/%m/%Y')
        df2.loc[row_num, 'Date ended'] = pd.to_datetime(input(f"When did you finish {row['Title_x']} ? dd/mm/YYYY "), format='%d/%m/%Y')
        df2.loc[row_num, 'Check'] = "OK"
    df2.to_excel('files/work_files/gr_work_files/gr_dates_input.xlsx', index=False)
    return df, df2

def expand_gr_reading_split(row, columns, col):
    """Splits the rows to have one row per day, with a division of the total pages in the book by the number of days to read it"""
    if (row['Date started'] != row['Date started']) | (row['Date ended'] != row['Date ended']):
        date_df = pd.DataFrame(columns=col)
        for col in col[:-2]:
            date_df[col] = [row[col]]
        date_df['Timestamp'] = [row['Date started']]
        date_df['page_split'] = [row['Number of Pages']]
        return date_df
    dates = pd.date_range(row['Date started'], row['Date ended'], freq='D')
    date_df = pd.DataFrame({'Timestamp': dates})
    for col in columns[:-2]:
        date_df[col] = row[col]
    date_df['page_split'] = row['Number of Pages'] / len(dates) if row['Number of Pages'] == row['Number of Pages'] else None
    return date_df


def get_genre(ISBN):
    """Function to retrieve the book genre from the Google API, but not used eventually"""
    api_url = 'https://www.googleapis.com/books/v1/volumes'
    params = {'q': f'isbn:{ISBN}'}
    time.sleep(5 )
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'items' in data and len(data['items']) > 0:
            book = data['items'][0]
            volume_info = book['volumeInfo']
            if 'categories' in volume_info:
                genres = volume_info['categories']
                return genres[0]
            else:
                return "issue1"
        else:
            return "issue2"
    else:
        return "issue3"

def create_gr_file():
    df, df2 = add_dates_read()
    df3 = pd.merge(df2[['Title', 'Book Id', 'Date started', 'Date ended', 'Check']], df, on='Book Id', how='left')\
            .rename(columns = {'Title_x' : 'Title', 'Bookshelves' : 'Genre'})
    df3['reading_duration'] = (df3['Date ended'] - df3['Date started']).dt.days + 1
    df3['Fiction_yn'] = df3['Genre'].apply(lambda x: "fiction" if x in fiction_genres else "non-fiction")
    columns = ['Book Id', 'Title', 'Author','Original Publication Year', 'My Rating',\
               'Average Rating','Genre','Fiction_yn','reading_duration','Number of Pages',\
               'Date started', 'Date ended']
    df_limited = df3[(df3['Exclusive Shelf']=='read') | (df3['Check']=='OK')][columns]
    col = columns[:-2]
    col.extend(['Timestamp', 'page_split'])
    expanded_df = pd.DataFrame(columns=col)
    for _, row in df_limited.iterrows():
        expanded_df = pd.concat([expanded_df, expand_gr_reading_split(row, columns, col)], ignore_index=True)
    expanded_df['Seconds'] = np.nan
    expanded_df['Fiction_yn'] = expanded_df['Genre'].apply(lambda x: "fiction" if x in fiction_genres else "non-fiction")
    expanded_df['Source'] = 'GoodReads'
    expanded_df['Title'] = expanded_df['Title'].apply(lambda x: str(x).strip())
    expanded_df.to_csv('files/processed_files/gr_processed.csv', sep = '|', index=False)

def process_gr_export():
    print('Starting the processing of the goodreads export \n')
    clean_rename_move_file("files/exports/goodreads_exports", "/Users/valen/Downloads", "goodreads_library_export.csv", "gr_export.csv")
    create_gr_file()
    print('gr_processed.csv was created \n')


#process_gr_export()
