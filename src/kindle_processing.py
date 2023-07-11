import pandas as pd
import json
from utils import time_difference_correction
import requests
import time
from bs4 import BeautifulSoup

def amazon_scraping(gr_df):
    list_ASIN = list(gr_df['ASIN'].unique())
    dict_path = 'files/work_files/kindle_ASIN_dictionnary.json'
    gr_df['start_timestamp'] = pd.to_datetime(gr_df['start_timestamp'])
    goodreads_titles = list(pd.read_csv('files/processed_files/gr_processed.csv', sep = '|').Title.unique())
    uncap_goodreads_titles = [str(t).lower() for t in goodreads_titles]
    with open(dict_path, 'r') as f:
        dict_ASIN = json.load(f)
    for ASIN in list_ASIN:
        time.sleep(2)
        if ASIN in dict_ASIN.keys():
            continue
        else:
            url = f"https://www.amazon.fr/s?k={ASIN}&__mk_fr_FR=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=MCGNNB0G7BY7&sprefix=b09g6wdz9j%2Caps%2C663&ref=nb_sb_noss"
            HEADERS = ({'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/113.0',
                        'Accept-Language' : 'en-us, en;q=0.5'})
            html = requests.get(url, headers = HEADERS)
            soup = BeautifulSoup(html.text)
            title = soup.find('span', class_='a-size-base-plus a-color-base a-text-normal').text.strip()
            cleaned_title = str(title.split('(')[0].strip()).lower()
            print(f"Amazon scraped is {cleaned_title}\n")
            print(f"Started reading on {gr_df[gr_df['ASIN']==ASIN]['start_timestamp'].dt.date} \n")
            if cleaned_title not in uncap_goodreads_titles:
                confirmation = input('This title is not in GR list. Is this the correct title? OK if yes, else NOK: ')
            else:
                cleaned_title = goodreads_titles[uncap_goodreads_titles.index(cleaned_title)]
                confirmation = 'OK'
                dict_ASIN[ASIN] = cleaned_title
            print('\n')
            if confirmation != 'OK':
                new_title = input('Input the actual title: ')
                while new_title not in goodreads_titles:
                    new_title = input('Not in GR. Input the actual title: ')
                confirmation == 'OK'
                dict_ASIN[ASIN] = new_title
                print('\n')
    with open(dict_path, 'w') as f:
        json.dump(dict_ASIN, f)
    return dict_ASIN

def kindle_ratio(df):
    dict_path = 'files/work_files/kindle_title_ratio_dictionnary.json'
    with open(dict_path, 'r') as f:
        dict_ratio = json.load(f)
    for _, row in df.iterrows():
        if (row['Title'] in dict_ratio.keys()) & (dict_ratio[row['Title']] == dict_ratio[row['Title']]):
            continue
        else:
            result = None if row['number_of_page_flips'] < 100 else row['number_of_page_flips']/row['Number of Pages']
            dict_ratio[row['Title']] = result
    with open(dict_path, 'w') as f:
        json.dump(dict_ratio, f)
    return dict_ratio

def kindle_page_ratio(gr_df):
    df = pd.read_csv('files/processed_files/gr_processed.csv', sep = '|')
    gr_df['uncapitalized_title'] = gr_df['Title'].apply(lambda x: str(x).lower())
    df['uncapitalized_title'] = df['Title'].apply(lambda x: x.lower())
    merged = pd.merge(gr_df, df[['Book Id', 'uncapitalized_title', 'Number of Pages']], on = 'uncapitalized_title', how = 'left')
    #merged = pd.merge(gr_df, df[['Book Id', 'uncapitalized_title']], on = 'uncapitalized_title', how = 'left')
    merged['kindle_page_ratio'] = merged['Title'].map(kindle_ratio(merged))
    return merged[['Book Id', 'Title', 'kindle_page_ratio']].drop_duplicates()

def row_expander_minutes(row):
    minute_diff = (row['end_timestamp'] - row['start_timestamp']).total_seconds()/60
    if minute_diff <=1:
        date_df = pd.DataFrame(columns=['ASIN', 'Book Id', 'Title', 'Timestamp', 'Seconds', 'page_split'])
        new_row = {'ASIN': row ['ASIN'], 'Book Id' : row['Book Id'], 'Title' : row['Title'],\
                   'Timestamp': row['start_timestamp'].floor('T'),'Seconds' : row["total_reading_millis"]/1000,\
                   'page_split' : row['actual_page_flip']}
        date_df = date_df.append(new_row, ignore_index=True)
        return date_df
    dates = pd.date_range(row['start_timestamp'], row['end_timestamp']- pd.Timedelta(minutes=1), freq='T')
    date_df = pd.DataFrame({'Timestamp': dates.floor('T')})
    date_df['ASIN'] = row['ASIN']
    date_df['Book Id'] = row['Book Id']
    date_df['Title'] = row['Title']
    date_df['page_split'] = row['actual_page_flip']/minute_diff
    for index, _ in date_df.iterrows():
        start_diff = (row['start_timestamp'] - row['start_timestamp'].floor('T'))\
                      .total_seconds()
        end_diff = (row['end_timestamp'] - row['end_timestamp'].floor('T'))\
                    .total_seconds()
        if index == 0:
            date_df.loc[index,'Seconds'] = 60- start_diff
        elif index == date_df.shape[0]-1:
            date_df.loc[index,'Seconds'] = end_diff
        else:
            date_df.loc[index,'Seconds'] = 60
    return date_df

def process_kindle_export():
    path = "files/exports/kindle_exports/Kindle.Devices.ReadingSession/Kindle.Devices.ReadingSession.csv"
    df = pd.read_csv(path)
    df = df[df['start_timestamp']!="Not Available"]
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])\
                            .apply(lambda x: x.to_pydatetime()\
                            .replace(tzinfo=None))\
                            .apply(lambda x: time_difference_correction(x,'GMT'))
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'])\
                            .apply(lambda x: x.to_pydatetime()\
                            .replace(tzinfo=None))\
                            .apply(lambda x: time_difference_correction(x,'GMT'))
    gr_df = df.groupby("ASIN").aggregate({'start_timestamp' : 'min', 'end_timestamp' : 'max', 'number_of_page_flips' : 'sum'}).reset_index()
    gr_df['start_timestamp'] = gr_df['start_timestamp'].dt.date
    dict_ASIN = amazon_scraping(gr_df)
    df['Title'] = df['ASIN'].map(dict_ASIN)
    gr_df['Title'] = gr_df['ASIN'].map(dict_ASIN)
    df = df[['Title','start_timestamp', 'end_timestamp', 'ASIN', 'total_reading_millis', 'number_of_page_flips']].sort_values('start_timestamp')
    df = pd.merge(df, kindle_page_ratio(gr_df), on = 'Title', how = 'left')
    df = df[df['kindle_page_ratio'].notna()]
    df['actual_page_flip'] = df['number_of_page_flips']/df['kindle_page_ratio']
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'])
    new_df = pd.DataFrame(columns=['ASIN', 'Book Id', 'Title', 'Timestamp', 'Seconds', 'page_split'])
    for _, row in df.iterrows():
        new_df = pd.concat([new_df, row_expander_minutes(row)], ignore_index=True)
    dict_new_ratio = {}
    for title in list(df.Title.unique()):
        dict_new_ratio[title] = df[df['Title'] == title].actual_page_flip.sum()/new_df[new_df['Title'] == title].page_split.sum()
    new_df['page_split'] = new_df.apply(lambda x: x.page_split * dict_new_ratio[x.Title], axis = 1)
    new_df['Source'] = 'Kindle'
    new_df.sort_values('Timestamp', ascending = False).to_csv('files/processed_files/kindle_processed.csv', sep = '|', index = False)

#process_kindle_export()
