import pandas as pd
import requests
import json
import os
import webbrowser
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def manually_change_book_cover(title_input=None, author_only=None):
    """
    Manually change the cover for a book based on user input.
    Provides option to use API search results or input a custom URL.

    Args:
        title_input (str, optional): Part of the book title to search for. If None, will prompt the user.
        author_only (bool, optional): If True, search by author only. Default is False.
    """


    # Get title input if not provided
    if title_input is None:
        title_input = input("Enter book title to search: ")

    # Handle author_only parameter
    if author_only is None:
        author_only_input = input("Do you want to make the search on author name only? Answer with True or False: ")
        author_only = author_only_input.lower() in ['true', 't', 'yes', 'y', '1']
    elif isinstance(author_only, str):
        author_only = author_only.lower() in ['true', 't', 'yes', 'y', '1']

    # First, load the Goodreads processed data
    print(f"Searching for books matching: '{title_input}'")
    try:
        df_gr = pd.read_csv('files/processed_files/gr_processed.csv', sep='|')
        # Get unique books to avoid duplicates
        df_unique = df_gr[['Book Id', 'Title', 'Author']].drop_duplicates()
    except Exception as e:
        print(f"Error loading Goodreads data: {str(e)}")
        return

    # Search for books matching the input (case insensitive)
    matches = df_unique[df_unique['Title'].str.contains(title_input, case=False, na=False)]

    if len(matches) == 0:
        print(f"No books found matching '{title_input}'")
        return

    # Display matches and let user select a book
    print(f"\nFound {len(matches)} matching books:")
    for i, (_, row) in enumerate(matches.iterrows(), 1):
        print(f"{i}. {row['Title']} by {row['Author']} (ID: {row['Book Id']})")

    # Let user select a book
    while True:
        try:
            choice = input("\nSelect a book number (or 'q' to quit): ")
            if choice.lower() == 'q':
                return

            choice = int(choice)
            if 1 <= choice <= len(matches):
                selected = matches.iloc[choice-1]
                break
            else:
                print(f"Please enter a number between 1 and {len(matches)}")
        except ValueError:
            print("Please enter a valid number")

    title = selected['Title']
    author = selected['Author']

    print(f"\nSelected book: '{title}' by {author}")

    # If author_only is True, use only the author name for search
    if author_only:
        print(f"\nSearching for covers using author only: '{author}'")
        query = f"inauthor:{author.replace(' ', '+')}"
    else:
        # Use the original title and author
        query = f"{title} {author}".replace(' ', '+')

    # Search for covers
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=10"

    try:
        response = requests.get(url)
        data = response.json()

        cover_options = []

        if 'items' in data and len(data['items']) > 0:
            # Filter and prepare cover options
            for item in data['items']:
                volume_info = item.get('volumeInfo', {})

                # Only include items with image links
                if 'imageLinks' in volume_info:
                    option = {
                        'id': item['id'],
                        'title': volume_info.get('title', 'Unknown'),
                        'authors': volume_info.get('authors', ['Unknown']),
                        'publisher': volume_info.get('publisher', 'Unknown'),
                        'cover_url': volume_info['imageLinks'].get('thumbnail')
                    }
                    cover_options.append(option)

            if cover_options:
                # Print the options with details
                print(f"\nFound {len(cover_options)} cover options:")

                for i, option in enumerate(cover_options, 1):
                    authors = ', '.join(option['authors'][:2])
                    if len(option['authors']) > 2:
                        authors += ' et al.'

                    print(f"\n{i}. {option['title']}")
                    print(f"   Authors: {authors}")
                    print(f"   Publisher: {option['publisher']}")
                    print(f"   Cover URL: {option['cover_url']}")

                # Display the covers in a compact way
                n = len(cover_options)
                cols = min(5, n)  # More columns for compactness
                rows = (n + cols - 1) // cols

                # Reduce the figure size and make the images smaller
                plt.figure(figsize=(12, max(4, 2 * rows)))

                for i, option in enumerate(cover_options):
                    try:
                        # Get the cover image
                        img_url = option['cover_url']
                        if not img_url:
                            continue

                        response = requests.get(img_url)
                        img = Image.open(BytesIO(response.content))

                        # Create subplot
                        plt.subplot(rows, cols, i+1)
                        plt.imshow(img)
                        plt.title(f"{i+1}", fontsize=10)  # Just show the number
                        plt.axis('off')
                    except Exception as e:
                        print(f"Error displaying option {i+1}: {str(e)}")

                plt.tight_layout()
                plt.subplots_adjust(hspace=0.5)  # Add more space between rows
                plt.show()

                # Let user select a cover or enter custom URL
                print("Select a cover number, enter 0 to provide your own URL, or 'q' to quit")
                choice = input("Your choice: ")

                if choice.lower() == 'q':
                    return

                choice = int(choice)

                # Option to provide custom URL
                if choice == 0:
                    print("\nPlease provide a direct URL to the cover image.")
                    print("Tip: Right-click on an image and select 'Open image in new tab' or 'View image',")
                    print("     then copy the URL from the address bar.")
                    # Open Goodreads in browser to help with search
                    goodreads_search_url = f"https://www.goodreads.com/search?q={title.replace(' ', '+')}+{author.replace(' ', '+')}"
                    print(f"\nOpening Goodreads search in your browser to help find covers...")
                    try:
                        webbrowser.open(goodreads_search_url)
                    except Exception as e:
                        print(f"Could not open browser: {str(e)}")

                    image_url = input("\nEnter image URL: ")

                    # Validate the URL
                    response = requests.get(image_url)
                    if response.status_code != 200:
                        print(f"Error: Could not access the URL (Status code: {response.status_code})")
                        return

                    # No need to display the image again, proceed with the custom URL
                    selected_cover = {
                        'cover_url': image_url
                    }
                elif 1 <= choice <= len(cover_options):
                    selected_cover = cover_options[choice-1]
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(cover_options)}")
                    return
            else:
                print("No cover images found from API search.")
                print("\nPlease provide a direct URL to the cover image.")
                print("Tip: Right-click on an image and select 'Open image in new tab' or 'View image',")
                print("     then copy the URL from the address bar.")

                image_url = input("\nEnter image URL: ")

                # Validate the URL
                response = requests.get(image_url)
                if response.status_code != 200:
                    print(f"Error: Could not access the URL (Status code: {response.status_code})")
                    return

                selected_cover = {
                    'cover_url': image_url
                }
        else:
            print("No cover options found from API search.")
            print("\nPlease provide a direct URL to the cover image.")
            print("Tip: Right-click on an image and select 'Open image in new tab' or 'View image',")
            print("     then copy the URL from the address bar.")

            image_url = input("\nEnter image URL: ")

            # Validate the URL
            response = requests.get(image_url)
            if response.status_code != 200:
                print(f"Error: Could not access the URL (Status code: {response.status_code})")
                return

            selected_cover = {
                'cover_url': image_url
            }

        # Load existing covers
        covers_dict_path = 'files/work_files/gr_work_files/book_covers.json'
        if os.path.exists(covers_dict_path):
            with open(covers_dict_path, 'r') as f:
                cover_dict = json.load(f)
        else:
            cover_dict = {}

        # Create book key
        book_key = f"{title}||{author}"

        # Update the cover dictionary
        cover_dict[book_key] = {
            'cover_url': selected_cover['cover_url'],
            'title': title,
            'author': author,
            'manually_selected': True
        }

        # Save the updated dictionary
        with open(covers_dict_path, 'w') as f:
            json.dump(cover_dict, f)

        print(f"\nCover updated successfully for '{title}'")

    except Exception as e:
        print(f"Error: {str(e)}")

manually_change_book_cover()
