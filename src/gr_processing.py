import pandas as pd
import numpy as np
from datetime import datetime
from utils import today_export
from drive_storage import update_file
import subprocess
import requests
import time

fiction_genres = ['drama', 'horror', 'thriller']

def add_dates_read():
    df = pd.read_csv("files/exports/goodreads_exports/gr_export.csv")
    df2 = pd.read_excel('files/work_files/gr_work_files/gr_dates_input.xlsx')
    df2["Book Id"].fillna(0, inplace=True)
    d = pd.merge(df, df2, on = "Book Id", how = 'left')
    d = d[(d['Exclusive Shelf'] == 'read') & (d['Check'].isna())].reset_index(drop = True)
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

def gr_only_bookid(df):
    groups = df.groupby(['Book Id'])
    gr_only_bookid = [g['Book Id'].values[0] for _, g in groups if all(g['Source'] == 'GoodReads')]
    return gr_only_bookid

def process_gr_export():
    df, df2 = add_dates_read()
    #df['Genre'] = df['ISBN'].apply(lambda x: get_genre(x))
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

def get_genre(ISBN):
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

def merge_gr_kindle():
    df_gr = pd.read_csv('files/processed_files/gr_processed.csv', sep ='|')
    df_kl = pd.read_csv('files/processed_files/kindle_processed.csv', sep = '|')
    columns = list(df_gr.columns[2:-4])
    columns.append('Book Id')
    enhanced_kl=pd.merge(df_kl, df_gr[columns], on = 'Book Id', how = 'left')
    all_data = pd.concat([df_gr, enhanced_kl], ignore_index = True)
    gr_only_df = all_data[all_data['Book Id'].isin(gr_only_bookid(all_data))]
    kl_only_df = all_data[all_data['Source'] == 'Kindle']
    cleaned_df = pd.concat([kl_only_df, gr_only_df], ignore_index=True).sort_values('Timestamp', ascending = False)
    cleaned_df['Timestamp'] = pd.to_datetime(cleaned_df['Timestamp'])
    cleaned_df.drop_duplicates(inplace = True)
    cleaned_df.to_csv('files/processed_files/kindle_gr_processed.csv', sep = '|', index = False, encoding = 'utf-16')

process_gr_export()
merge_gr_kindle()
#update_file('processed_files/kindle_gr_processed.csv')
