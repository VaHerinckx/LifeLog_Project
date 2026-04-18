"""
Genre Mapping Utility

Maps specific Spotify genre_1 values to simplified broad categories.
Loads mappings from CSV file for easy maintenance.

Usage:
    from topic_processing.music.genre_mapping import get_simplified_genre
    df['simplified_genre'] = df['genre_1'].apply(get_simplified_genre)

    # Run directly to analyze unmapped genres:
    python src/topic_processing/music/genre_mapping.py
"""

import os
import sys
import pandas as pd

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# Global variable to store loaded mappings
GENRE_MAPPING = {}
MAPPING_FILE = os.path.join(project_root, 'files', 'work_files', 'lastfm_work_files', 'genre_mapping.csv')
from src.utils.logger import log


def load_genre_mapping():
    """
    Load genre mappings from CSV file.

    Returns:
        dict: Genre mapping dictionary {specific_genre: broad_category}
    """
    global GENRE_MAPPING

    if GENRE_MAPPING:
        # Already loaded
        return GENRE_MAPPING

    try:
        df = pd.read_csv(MAPPING_FILE, sep='|', encoding='utf-8')

        # Validate CSV structure
        if not all(col in df.columns for col in ['specific_genre', 'broad_category']):
            raise ValueError("CSV must have 'specific_genre' and 'broad_category' columns")

        # Build dictionary (specific_genre -> broad_category)
        GENRE_MAPPING = dict(zip(
            df['specific_genre'].str.lower().str.strip(),
            df['broad_category']
        ))

        return GENRE_MAPPING

    except FileNotFoundError:
        log.error(f"Error: Genre mapping file not found: {MAPPING_FILE}")
        log.info(f"Create this file with format: specific_genre|broad_category")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error loading genre mapping: {e}")
        sys.exit(1)


def get_simplified_genre(genre_1):
    """
    Map specific genre_1 to simplified genre category.

    Uses hierarchical matching:
    1. Direct mapping (exact match)
    2. Partial matching for compound genres
    3. Fallback to 'Other' if unmapped

    Args:
        genre_1: Specific genre from Spotify API (e.g., "chicago rap")

    Returns:
        Simplified genre category (e.g., "Hip Hop") or "Unknown"/"Other"

    Examples:
        >>> get_simplified_genre('chicago rap')
        'Hip Hop'
        >>> get_simplified_genre('chamber pop')
        'Pop'
        >>> get_simplified_genre(None)
        'Unknown'
    """
    # Load mappings if not already loaded
    if not GENRE_MAPPING:
        load_genre_mapping()

    if not genre_1 or pd.isna(genre_1) or genre_1 == '':
        return 'Unknown'

    genre_lower = str(genre_1).lower().strip()

    # Direct mapping (exact match)
    if genre_lower in GENRE_MAPPING:
        return GENRE_MAPPING[genre_lower]

    # Partial matching for compound genres
    # Example: "australian indie folk" contains "indie folk" → "Indie"
    for key, value in GENRE_MAPPING.items():
        if key in genre_lower:
            return value

    # Unmapped genre
    return 'Other'


def analyze_unmapped_genres(df, top_n=50):
    """
    Analyze unmapped genres and print LLM-ready prompt.

    Displays:
    - Genre mapping coverage statistics
    - Top N unmapped genres by frequency
    - Copy-paste ready prompt for LLM categorization

    Args:
        df: DataFrame with 'genre_1' column
        top_n: Number of top unmapped genres to display (default: 50)
    """
    # Load mappings if not already loaded
    if not GENRE_MAPPING:
        load_genre_mapping()

    # Add temporary simplified column
    df_temp = df.copy()
    df_temp['_simplified'] = df_temp['genre_1'].apply(get_simplified_genre)

    # Filter to only unmapped ('Other')
    unmapped = df_temp[df_temp['_simplified'] == 'Other']['genre_1']

    # Calculate coverage statistics
    total_tracks = len(df_temp)
    unmapped_tracks = len(df_temp[df_temp['_simplified'] == 'Other'])
    mapped_tracks = total_tracks - unmapped_tracks
    coverage_pct = (mapped_tracks / total_tracks) * 100 if total_tracks > 0 else 0

    # Get unique categories
    unique_categories = sorted(set(GENRE_MAPPING.values()))

    # Print statistics header
    log.info("\n" + "="*60)
    log.progress("Genre Mapping Statistics")
    log.info("="*60)
    log.success(f"Loaded {len(GENRE_MAPPING)} mappings | {len(unique_categories)} categories | {coverage_pct:.1f}% coverage")
    log.info(f"Mapped: {mapped_tracks:,} tracks | Unmapped: {unmapped_tracks:,} tracks")
    log.progress("\n Current Categories:")
    log.info(", ".join(unique_categories))

    if len(unmapped) == 0:
        log.success("\n All genres are mapped!")
        return

    # Count unmapped genre occurrences
    genre_counts = unmapped.value_counts().head(top_n)

    # Print unmapped genres
    log.progress(f"\n Top {min(top_n, len(genre_counts))} Unmapped Genres (currently mapped to 'Other'):")
    log.info("="*60)
    for genre, count in genre_counts.items():
        log.info(f"{genre:45} {count:>10,}")

    # Print LLM-ready prompt
    log.info("\n" + "━"*60)
    log.info("PROMPT FOR LLM (copy below this line)")
    log.info("━"*60)
    log.info("\nI have these unmapped music genres that need categorization:\n")

    # Format genres for LLM
    for genre, count in genre_counts.items():
        log.info(f"- {genre} ({count:,} tracks)")

    log.info(f"\nPlease map each genre to ONE of these broad categories:")
    log.info(f"- {', '.join(unique_categories)}")

    log.info("\nProvide the mappings in this CSV format (pipe-delimited):")
    log.info("specific_genre|broad_category")

    log.info("\nExample:")
    if len(genre_counts) > 0:
        first_genre = genre_counts.index[0]
        log.info(f"{first_genre}|[Category]")

    log.info("\nAfter you provide the mappings, append them to files/work_files/lastfm_work_files/genre_mapping.csv")
    log.info("━"*60)
    log.blank()


def get_mapping_stats():
    """
    Returns statistics about the current genre mapping.

    Returns:
        dict: Statistics including total mappings and categories
    """
    # Load mappings if not already loaded
    if not GENRE_MAPPING:
        load_genre_mapping()

    unique_categories = set(GENRE_MAPPING.values())

    return {
        'total_mappings': len(GENRE_MAPPING),
        'unique_categories': len(unique_categories),
        'categories': sorted(unique_categories)
    }


if __name__ == '__main__':
    # Display mapping statistics and analyze unmapped genres
    log.milestone("\n Loading genre mappings from CSV...")

    try:
        load_genre_mapping()
        stats = get_mapping_stats()

        log.success(f"Successfully loaded {stats['total_mappings']} genre mappings")
        log.info(f"Categories: {', '.join(stats['categories'])}")

        # Show example mappings
        log.info("\n Example mappings:")
        examples = [
            ('chicago rap', get_simplified_genre('chicago rap')),
            ('chamber pop', get_simplified_genre('chamber pop')),
            ('indie folk', get_simplified_genre('indie folk')),
            ('chillwave', get_simplified_genre('chillwave')),
            ('unknown genre', get_simplified_genre('unknown genre'))
        ]
        for example_genre, result in examples:
            log.info(f"{example_genre:30}  {result}")

        # Load music data to analyze unmapped genres
        log.progress("\n Analyzing unmapped genres from processed data...")

        # Path to processed music data (in topic_processed_files now)
        processed_file = os.path.join(
            project_root,
            'files',
            'topic_processed_files',
            'music',
            'music_processed.csv'
        )

        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file, sep='|', encoding='utf-8')

            if 'genre_1' in df.columns:
                analyze_unmapped_genres(df, top_n=1000)
            else:
                log.warning("Warning: 'genre_1'column not found in processed data")
                log.info(" Run the Music topic coordinator pipeline first to generate data")
        else:
            log.warning(f"Warning: Processed file not found: {processed_file}")
            log.info(" Run the Music topic coordinator pipeline first to generate data")
            log.info("\n To generate data, run:")
            log.info(" python src/topic_processing/music/music_processing.py")

    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)
