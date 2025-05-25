import pandas as pd
import numpy as np
import json
from src.utils.utils_functions import time_difference_correction, find_unzip_folder, clean_rename_move_folder
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
import os
from dotenv import load_dotenv
from src.utils.utils_functions import get_response
from openai import OpenAI
load_dotenv()
api_key = os.environ['OpenAI_Key']

dict_path = 'files/work_files/kindle_ASIN_dictionnary.json'

def prompt_book_title(client, book_title):
    """Generates a ChatGPT prompt to get correct title out of Amazon scraped """
    system_prompt = """You are a helpful librarian that is knowledgeable on all types of books.
    You can speak French, English and Dutch. You are very laconic and only answer the question asked and nothing else"""
    user_prompt = f"""I am trying to retrieve book titles by scraping Amazon on their ASIN
    numbers, but the result sometimes contains the title along with a promotional message or the edition of the book.
    Please me to extract the book title from the result delimited by triple backticks ```{book_title}```.
    Please just answer with only the title and nothing else.
    Below are examples of how your answer should look like (don't include them in your answer):
    1. Input: "Atonement, the no1 best seller" Your answer : "Atonement"
    2. Input: "The girl with all the Gifts: The most Original thriller you will read this year" Your answer : "The girl with all the Gifts"
    """
    return get_response(client, system_prompt, user_prompt)

def amazon_scraping_gpt_call(gr_df):
    """Scrapes Amazon to find the titles of Books based on their ASIN number using Selenium webdriver
    If title is not in Goodreads list, GPT 3.5 extracts it."""
    list_ASIN = list(gr_df['ASIN'].unique())
    dict_path = 'files/work_files/kindle_ASIN_dictionnary.json'
    gr_df['start_timestamp'] = pd.to_datetime(gr_df['start_timestamp'])
    goodreads_titles = list(pd.read_csv('files/processed_files/gr_processed.csv', sep = '|').Title.unique())
    uncap_goodreads_titles = [str(t).lower() for t in goodreads_titles]
    with open(dict_path, 'r') as f:
        dict_ASIN = json.load(f)
    new_ASIN = [asin for asin in list_ASIN if asin not in dict_ASIN.keys()]
    if len(new_ASIN) == 0:
        return "No new ASIN"
    client = OpenAI(api_key = api_key)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_service = ChromeService('files/other_files/chromedriver_mac64/chromedriver')
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    #driver = webdriver.Chrome('files/other_files/chromedriver_mac64(2)/chromedriver',options=chrome_options)
    for ASIN in new_ASIN:
        time.sleep(2)
        if ASIN in dict_ASIN.keys():
            continue
        else:
            url = f"https://www.amazon.fr/s?k={ASIN}&__mk_fr_FR=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=MCGNNB0G7BY7&sprefix=b09g6wdz9j%2Caps%2C663&ref=nb_sb_noss"
            driver.get(url)
            time.sleep(5)
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'lxml')
            title_element = soup.find('span', class_='a-size-base-plus a-color-base a-text-normal')
            if title_element is None:
                print("Title element not found.")
                continue
            title = title_element.text.strip()
            cleaned_title = str(title.split('(')[0].strip()).lower()
            print(f"Amazon scraped is {cleaned_title}\n")
            if cleaned_title in uncap_goodreads_titles:
                cleaned_title = goodreads_titles[uncap_goodreads_titles.index(cleaned_title)]
                dict_ASIN[ASIN] = cleaned_title
            else:
                new_title = prompt_book_title(client, cleaned_title)
                cleaned_new_title = str(new_title.split('(')[0].strip()).lower()
                if cleaned_new_title not in uncap_goodreads_titles:
                    new_title = input(f'GPT says title is {new_title}. It is not in GR. Input the actual title: ')
                    dict_ASIN[ASIN] = new_title
                else:
                    dict_ASIN[ASIN] = goodreads_titles[uncap_goodreads_titles.index(cleaned_new_title)]
                print('\n')
    with open(dict_path, 'w') as f:
        json.dump(dict_ASIN, f)
    driver.quit()


def amazon_scraping_sel(gr_df):
    """Scrapes Amazon to find the titles of Books based on their ASIN number using Selenium webdriver"""
    list_ASIN = list(gr_df['ASIN'].unique())
    dict_path = 'files/work_files/kindle_ASIN_dictionnary.json'
    gr_df['start_timestamp'] = pd.to_datetime(gr_df['start_timestamp'])
    goodreads_titles = list(pd.read_csv('files/processed_files/gr_processed.csv', sep = '|').Title.unique())
    uncap_goodreads_titles = [str(t).lower() for t in goodreads_titles]
    with open(dict_path, 'r') as f:
        dict_ASIN = json.load(f)
    new_ASIN = [asin for asin in list_ASIN if asin not in dict_ASIN.keys()]
    if len(new_ASIN) == 0:
        return "No new ASIN"
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_service = ChromeService('files/other_files/chromedriver_mac64/chromedriver')
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    #driver = webdriver.Chrome('files/other_files/chromedriver_mac64(2)/chromedriver',options=chrome_options)
    for ASIN in new_ASIN:
        time.sleep(2)
        if ASIN in dict_ASIN.keys():
            continue
        else:
            url = f"https://www.amazon.fr/s?k={ASIN}&__mk_fr_FR=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=MCGNNB0G7BY7&sprefix=b09g6wdz9j%2Caps%2C663&ref=nb_sb_noss"
            driver.get(url)
            time.sleep(5)
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'lxml')
            title_element = soup.find('span', class_='a-size-base-plus a-color-base a-text-normal')
            if title_element is None:
                print("Title element not found.")
                continue
            title = title_element.text.strip()
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
    driver.quit()

def kindle_ratio(df):
    """Computes the ratio of Kindle pages vs. Goodreads pages, as Kindle has a different way of calculating pages"""
    dict_path = 'files/work_files/kindle_title_ratio_dictionnary.json'
    with open(dict_path, 'r') as f:
        dict_ratio = json.load(f)
    for _, row in df.iterrows():
        if row['Title'] not in dict_ratio.keys():
            result = None if row['number_of_page_flips'] < 100 else row['number_of_page_flips']/row['Number of Pages']
            dict_ratio[row['Title']] = result
        elif dict_ratio[row['Title']] != dict_ratio[row['Title']]:
            result = None if row['number_of_page_flips'] < 100 else row['number_of_page_flips']/row['Number of Pages']
            dict_ratio[row['Title']] = result
        else:
            continue
    with open(dict_path, 'w') as f:
        json.dump(dict_ratio, f)
    return dict_ratio

def kindle_page_ratio(gr_df):
    """Computes the ratio of Kindle pages vs. Goodreads pages, as Kindle has a different way of calculating pages"""
    df = pd.read_csv('files/processed_files/gr_processed.csv', sep = '|')
    gr_df['uncapitalized_title'] = gr_df['Title'].apply(lambda x: str(x).lower())
    df['uncapitalized_title'] = df['Title'].apply(lambda x: x.lower())
    merged = pd.merge(gr_df, df[['Book Id', 'uncapitalized_title', 'Number of Pages']], on = 'uncapitalized_title', how = 'left')
    #merged = pd.merge(gr_df, df[['Book Id', 'uncapitalized_title']], on = 'uncapitalized_title', how = 'left')
    merged['kindle_page_ratio'] = merged['Title'].map(kindle_ratio(merged))
    return merged[['Book Id', 'Title', 'kindle_page_ratio']].drop_duplicates()

def row_expander_minutes(row):
    """Expands the rows to have one per minute, rather than aggregated per reading session"""
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

def session_split(df, index):
    """Function that splits the different reading sessions in the data"""
    try:
        value = int((df["Timestamp"].iloc[index] - df["Timestamp"].iloc[index + 1]).seconds > 60)
    except IndexError:
        value = 1
    return value

def session_duration(df, rownum):
    """Function that computes the duration of each reading session"""
    new_df = df[df["RowNum"] >= rownum]
    session_duration = 0
    if rownum == 0:
        for _, row in new_df.iterrows():
            if row["NewSession"] == 0:
                session_duration += row["Seconds"]
            else:
                break
    elif (df[df["RowNum"] == rownum]["NewSession"][rownum] == 0) & (df[df["RowNum"] == rownum - 1]["NewSession"][rownum - 1] == 1):
        for _, row in new_df.iterrows():
            if row["NewSession"] == 0:
                session_duration += row["Seconds"]
            else:
                break
    return session_duration/60 if session_duration > 0 else np.nan

def create_kindle_file():
    """Process the kindle export by doing all kinds of reformatting"""
    path = "files/exports/kindle_exports/Kindle.Devices.ReadingSession/Kindle.Devices.ReadingSession.csv"
    df = pd.read_csv(path)
    df = df[df['start_timestamp']!="Not Available"]
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], utc = True)\
                            .apply(lambda x: time_difference_correction(x,'GMT'))
                            #.apply(lambda x: x.to_pydatetime()\
                            #.replace(tzinfo=None))\
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'], utc = True)\
                            .apply(lambda x: time_difference_correction(x,'GMT'))
                            #.apply(lambda x: x.to_pydatetime()\
                            #.replace(tzinfo=None))\
    gr_df = df.groupby("ASIN").aggregate({'start_timestamp' : 'min', 'end_timestamp' : 'max', 'number_of_page_flips' : 'sum'}).reset_index()
    gr_df['start_timestamp'] = gr_df['start_timestamp'].dt.date
    #amazon_scraping_sel(gr_df)
    amazon_scraping_gpt_call(gr_df)
    with open(dict_path, 'r') as f:
        dict_ASIN = json.load(f)
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
    new_df.sort_values('Timestamp', ascending = False, inplace = True)
    new_df.reset_index(drop=True, inplace = True)
    new_df = new_df.reset_index().rename(columns = {"index" : "RowNum"})
    new_df['NewSession'] = new_df.apply(lambda x: session_split(new_df, x['RowNum']), axis = 1)
    new_df['SessionDuration'] = new_df.apply(lambda x: session_duration(new_df, x['RowNum']), axis = 1)
    new_df['Source'] = 'Kindle'
    new_df.drop(columns = "RowNum", axis = 1).to_csv('files/processed_files/kindle_processed.csv', sep = '|', index = False)

def process_kindle_export():
    print('Starting the processing of the Kindle export \n')
    find_unzip_folder("kindle")
    clean_rename_move_folder("files/exports", "/Users/valen/Downloads", "kindle_export_unzipped", "kindle_exports")
    create_kindle_file()
    print('kindle_processed.csv was created\n')
