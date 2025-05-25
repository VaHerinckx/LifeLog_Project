import pandas as pd
import numpy as np
import requests
import json
import os
import time
from pathlib import Path

def get_book_covers(df):
    """
    Retrieves book cover images from Google Books API and stores the URLs
    in a local JSON file for future reference.
    """
    # Define paths
    covers_dict_path = 'files/work_files/gr_work_files/book_covers.json'

    # Load existing cover data if available
    if os.path.exists(covers_dict_path):
        with open(covers_dict_path, 'r') as f:
            cover_dict = json.load(f)
    else:
        cover_dict = {}

    # Get unique books
    unique_books = df[['Title', 'Author']].drop_duplicates()
    print(f"Trying to retrieve book cover for {len(unique_books)} unique book(s)")

    # Counter for API tracking
    new_covers_count = 0

    # Create a key for lookups based on title and author
    for _, row in unique_books.iterrows():
        title = str(row['Title']).strip()
        author = str(row['Author']).strip() if 'Author' in row and row['Author'] == row['Author'] else ""

        # Create a unique key for the book
        book_key = f"{title}||{author}"

        # Skip if we already have this book's cover
        if book_key in cover_dict:
            continue

        # Sleep to avoid hitting API rate limits
        time.sleep(0.5)

        try:
            # Query Google Books API - requesting multiple results for scoring
            query = f"{title} {author}".replace(' ', '+')
            response = requests.get(
                f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=5"
            )

            if response.status_code == 200:
                data = response.json()

                # Extract cover URL if available
                if 'items' in data and len(data['items']) > 0:
                    # Score the candidates to find the best edition
                    candidates = []

                    for item in data['items']:
                        volume_info = item.get('volumeInfo', {})

                        # Skip items without image links
                        if 'imageLinks' not in volume_info:
                            continue

                        # Create candidate with scoring
                        candidate = {
                            'score': 0,
                            'volume_info': volume_info
                        }

                        # 1. English language preference (major boost)
                        if volume_info.get('language') == 'en':
                            candidate['score'] += 50

                        # 2. Rating boost
                        if 'averageRating' in volume_info:
                            candidate['score'] += min(volume_info['averageRating'] * 5, 20)

                        # 3. Popularity boost (number of ratings)
                        if 'ratingsCount' in volume_info:
                            import math
                            candidate['score'] += min(math.log(volume_info['ratingsCount'] + 1) * 3, 20)

                        # Add to candidates
                        candidates.append(candidate)

                    # Get the best candidate
                    if candidates:
                        candidates.sort(key=lambda x: x['score'], reverse=True)
                        best_candidate = candidates[0]
                        volume_info = best_candidate['volume_info']

                        # Get the best image available
                        if 'large' in volume_info['imageLinks']:
                            cover_url = volume_info['imageLinks']['large']
                        elif 'medium' in volume_info['imageLinks']:
                            cover_url = volume_info['imageLinks']['medium']
                        else:
                            cover_url = volume_info['imageLinks'].get('thumbnail',
                                     volume_info['imageLinks'].get('smallThumbnail', None))

                        if cover_url:
                            # Store in dictionary
                            cover_dict[book_key] = {
                                'cover_url': cover_url,
                                'title': title,
                                'author': author
                            }
                            new_covers_count += 1

                            # Save periodically to avoid losing progress
                            if new_covers_count % 10 == 0:
                                with open(covers_dict_path, 'w') as f:
                                    json.dump(cover_dict, f)
                                print(f"Saved {new_covers_count} new covers")
            else:
                print(f"API error for {title}: {response.status_code}")

        except Exception as e:
            print(f"Error processing {title}: {str(e)}")

    # Save the final dictionary
    with open(covers_dict_path, 'w') as f:
        json.dump(cover_dict, f)

    print(f"Added {new_covers_count} new book covers. Total covers: {len(cover_dict)}")
    return cover_dict

def add_covers_to_dataframe(df):
    """
    Adds cover URLs to the book dataframe and saves an updated CSV file
    """
    # Load cover dictionary
    covers_dict_path = 'files/work_files/gr_work_files/book_covers.json'
    with open(covers_dict_path, 'r') as f:
        cover_dict = json.load(f)

    # Function to look up cover URL
    def get_cover_url(row):
        title = str(row['Title']).strip()
        author = str(row['Author']).strip() if 'Author' in row and row['Author'] == row['Author'] else ""
        book_key = f"{title}||{author}"

        if book_key in cover_dict:
            return cover_dict[book_key]['cover_url']
        return None

    # Add cover URL column
    df['cover_url'] = df.apply(get_cover_url, axis=1)
    print(f"Added cover URLs to {len(df)} book records")

    return df

def gr_only_bookid(df):
    """Returns the books that are only in GoodReads, not in the Kindle data"""
    groups = df.groupby(['Book Id'])
    gr_only_bookid = [g['Book Id'].values[0] for _, g in groups if all(g['Source'] == 'GoodReads')]
    return gr_only_bookid


def flag_clicks(row, gr_date_df):
    """Add a flag to the rows where the date a book has been read is higher than what was manually input in GoodReads"""
    gr_date_df = gr_date_df[gr_date_df["Date ended"].notna()] #Condition added because it was failing when going through books without 'Date ended'
    if (not gr_date_df.loc[gr_date_df['Title'] == row["Title"]].empty) & (not gr_date_df.loc[gr_date_df['Title'] == row["Title"]]["Date ended"].empty):
        if row.Timestamp.date() > gr_date_df.loc[gr_date_df['Title'] == row["Title"]]['Date ended'].iloc[0].date():
            return 1
        else:
            return 0
    return 0

def remove_accidental_clicks(df):
    """Removes the data points where a book was accidentally clicked in the Kindle, to make sure reading dates remain coherent with what's in GR"""
    gr_date_df = pd.read_excel('files/work_files/gr_work_files/gr_dates_input.xlsx')
    df['Flag_remove'] = df.apply(lambda x: flag_clicks(x, gr_date_df), axis = 1)
    return df[df['Flag_remove'] == 0].drop('Flag_remove', axis = 1)

def duration(df, time, title, rowNum):
    """Computes the reading duration one time per title, making calculations in the PBI easier"""
    max_time = df[df["Title"] == title].Timestamp.max()
    max_row = df[(df["Title"] == title) & (df["Timestamp"] == max_time)].rowNum.min()
    if (time == max_time) & (rowNum == max_row):
        return df[df["Title"] == title].reading_duration.max()
    else:
        return np.nan

def produce_book_file():
    """Merges goodreads & kindle files"""
    df_gr = pd.read_csv('files/processed_files/gr_processed.csv', sep ='|')
    df_kl = pd.read_csv('files/processed_files/kindle_processed.csv', sep = '|')
    columns = list(df_gr.columns[2:-4])
    columns.append('Book Id')
    enhanced_kl = pd.merge(df_kl, df_gr[columns], on = 'Book Id', how = 'left')
    all_data = pd.concat([df_gr, enhanced_kl], ignore_index = True)
    gr_only_df = all_data[all_data['Book Id'].isin(gr_only_bookid(all_data))]
    kl_only_df = all_data[all_data['Source'] == 'Kindle']
    cleaned_df = pd.concat([kl_only_df, gr_only_df], ignore_index=True).sort_values('Timestamp', ascending = False)
    cleaned_df['Timestamp'] = pd.to_datetime(cleaned_df['Timestamp'], utc=True)
    cleaned_df = remove_accidental_clicks(cleaned_df)
    cleaned_df.drop_duplicates(inplace = True)
    cleaned_df = cleaned_df.reset_index().rename(columns = {"index" : "rowNum"})
    cleaned_df["reading_duration"] = cleaned_df.apply(lambda x: duration(cleaned_df, x.Timestamp, x.Title, x.rowNum), axis = 1)
    get_book_covers(cleaned_df)
    cleaned_df = add_covers_to_dataframe(cleaned_df)
    cleaned_df.drop(columns = "rowNum").to_csv('files/processed_files/kindle_gr_processed.csv', sep = '|', index = False, encoding = 'utf-16')


def process_book_exports():
    print('Merging the Kindle & Goodreads processed files \n')
    produce_book_file()


#process_book_exports("Y")
