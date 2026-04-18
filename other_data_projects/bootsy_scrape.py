#!/usr/bin/env python3
"""
Google Website Snippet Analyzer
Automatically extract Google's auto-generated snippets for all pages of a website

Setup:
1. Get your Google Custom Search API key and Search Engine ID (see setup guide)
2. Configure the settings below
3. Run the script: python google_snippet_analyzer.py

Requirements:
pip install requests pandas
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import os
import sys
from urllib.parse import urlparse

# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================================================

# Google API Credentials (REQUIRED)
GOOGLE_API_KEY = "AIzaSyCGsDrkTivc15aF9BXi1tUa3C2-gGYK3d8"  # Replace with your actual API key
SEARCH_ENGINE_ID = "1278c5fabb7194a4b"  # Replace with your actual Search Engine ID

# Website to analyze (REQUIRED)
TARGET_WEBSITE = "www.mg-united.eu"  # Replace with your website domain

# Output settings
OUTPUT_CSV_PATH = "google_snippets_analysis.csv"  # Path where CSV will be saved
OUTPUT_JSON_PATH = "google_snippets_raw_data.json"  # Optional: raw JSON backup

# Search settings
MAX_RESULTS_PER_SEARCH = 100  # Maximum pages to analyze (Google API limit: 100)
DELAY_BETWEEN_REQUESTS = 0.1  # Seconds to wait between API calls (be nice to Google)
INCLUDE_CACHE_URLS = True  # Include Google cache URLs in results
INCLUDE_STRUCTURED_DATA = True  # Include structured data information

# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class GoogleWebsiteSnippetAnalyzer:
    def __init__(self, api_key: str, search_engine_id: str):
        """Initialize the analyzer with Google API credentials"""
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Google-Snippet-Analyzer/1.0'
        })

    def analyze_website(self, website_url: str, max_results: int = 100) -> List[Dict]:
        """
        Analyze all pages of a website and extract Google snippets

        Args:
            website_url: The website domain to analyze
            max_results: Maximum number of pages to analyze

        Returns:
            List of dictionaries containing page data
        """
        # Clean up website URL to get domain
        domain = self._extract_domain(website_url)
        print(f"🔍 Analyzing website: {domain}")
        print(f"📊 Maximum results: {max_results}")
        print(f"⏱️  Estimated time: {max_results * DELAY_BETWEEN_REQUESTS:.1f} seconds")
        print("-" * 60)

        # Search for all pages on the domain
        query = f"site:{domain}"
        all_results = []

        # Paginate through results (Google returns max 10 per page)
        for start_index in range(1, min(max_results + 1, 101), 10):
            print(f"📖 Fetching results {start_index}-{min(start_index + 9, max_results)}...")

            page_results = self._search_page(query, start_index)

            if not page_results:
                print("❌ No more results found")
                break

            all_results.extend(page_results)

            # Rate limiting
            if DELAY_BETWEEN_REQUESTS > 0:
                time.sleep(DELAY_BETWEEN_REQUESTS)

            # Check if we've reached the desired number of results
            if len(all_results) >= max_results:
                all_results = all_results[:max_results]
                break

        print(f"✅ Analysis complete! Found {len(all_results)} pages")
        return all_results

    def _search_page(self, query: str, start_index: int) -> List[Dict]:
        """Perform a single page search and extract data"""
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'start': start_index,
            'num': 10
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if 'error' in data:
                error_msg = data['error'].get('message', 'Unknown API error')
                print(f"❌ API Error: {error_msg}")
                return []

            # Extract data from each search result
            results = []
            if 'items' in data:
                for i, item in enumerate(data['items'], start_index):
                    result = self._extract_page_data(item, i)
                    results.append(result)

            return results

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse API response: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return []

    def _extract_page_data(self, item: Dict, position: int) -> Dict:
        """Extract relevant data from a single search result"""
        # Basic page information
        result = {
            'position': position,
            'title': item.get('title', ''),
            'url': item.get('link', ''),
            'google_snippet': item.get('snippet', ''),
            'display_url': item.get('displayLink', ''),
            'formatted_url': item.get('formattedUrl', ''),
            'snippet_length': len(item.get('snippet', '')),
            'extracted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Add cache URL if available and requested
        if INCLUDE_CACHE_URLS and 'cacheId' in item:
            result['cache_id'] = item['cacheId']
            result['cache_url'] = f"https://webcache.googleusercontent.com/search?q=cache:{item['cacheId']}"
        else:
            result['cache_id'] = ''
            result['cache_url'] = ''

        # Extract structured data information
        if INCLUDE_STRUCTURED_DATA and 'pagemap' in item:
            pagemap = item['pagemap']

            # Common structured data types
            result['has_structured_data'] = 'Yes'
            result['structured_data_types'] = ', '.join(pagemap.keys())

            # Extract specific structured data
            result['meta_description'] = self._get_nested_value(pagemap, 'metatags', 0, 'description', '')
            result['meta_keywords'] = self._get_nested_value(pagemap, 'metatags', 0, 'keywords', '')
            result['og_title'] = self._get_nested_value(pagemap, 'metatags', 0, 'og:title', '')
            result['og_description'] = self._get_nested_value(pagemap, 'metatags', 0, 'og:description', '')
            result['og_image'] = self._get_nested_value(pagemap, 'metatags', 0, 'og:image', '')
            result['twitter_title'] = self._get_nested_value(pagemap, 'metatags', 0, 'twitter:title', '')
            result['twitter_description'] = self._get_nested_value(pagemap, 'metatags', 0, 'twitter:description', '')
            result['canonical_url'] = self._get_nested_value(pagemap, 'metatags', 0, 'canonical', '')

            # Schema.org data
            if 'person' in pagemap:
                result['schema_person'] = 'Yes'
            if 'organization' in pagemap:
                result['schema_organization'] = 'Yes'
            if 'article' in pagemap:
                result['schema_article'] = 'Yes'
            if 'product' in pagemap:
                result['schema_product'] = 'Yes'

        else:
            # Fill in empty structured data fields
            result['has_structured_data'] = 'No'
            result['structured_data_types'] = ''
            result['meta_description'] = ''
            result['meta_keywords'] = ''
            result['og_title'] = ''
            result['og_description'] = ''
            result['og_image'] = ''
            result['twitter_title'] = ''
            result['twitter_description'] = ''
            result['canonical_url'] = ''
            result['schema_person'] = 'No'
            result['schema_organization'] = 'No'
            result['schema_article'] = 'No'
            result['schema_product'] = 'No'

        return result

    def _get_nested_value(self, data: Dict, *keys, default=''):
        """Safely extract nested values from pagemap data"""
        try:
            current = data
            for key in keys[:-1]:
                if isinstance(key, int):
                    current = current[keys[keys.index(key)-1]][key]
                else:
                    current = current[key]

            final_key = keys[-1]
            if isinstance(current, list) and len(current) > 0:
                return current[0].get(final_key, default)
            elif isinstance(current, dict):
                return current.get(final_key, default)
            else:
                return default
        except (KeyError, IndexError, TypeError):
            return default

    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return url.replace('https://', '').replace('http://', '').split('/')[0]

    def save_to_csv(self, data: List[Dict], filepath: str) -> bool:
        """Save results to CSV file"""
        try:
            df = pd.DataFrame(data)

            # Reorder columns for better readability
            preferred_order = [
                'position', 'title', 'url', 'google_snippet', 'snippet_length',
                'meta_description', 'og_title', 'og_description',
                'display_url', 'has_structured_data', 'structured_data_types',
                'meta_keywords', 'og_image', 'twitter_title', 'twitter_description',
                'canonical_url', 'schema_person', 'schema_organization',
                'schema_article', 'schema_product', 'cache_url', 'extracted_at'
            ]

            # Only include columns that exist in the data
            available_columns = [col for col in preferred_order if col in df.columns]
            remaining_columns = [col for col in df.columns if col not in preferred_order]
            final_columns = available_columns + remaining_columns

            df = df[final_columns]

            # Save to CSV
            df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"💾 Results saved to CSV: {filepath}")
            print(f"📊 Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return True

        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            return False

    def save_to_json(self, data: List[Dict], filepath: str) -> bool:
        """Save raw results to JSON file for backup"""
        try:
            output_data = {
                'website': TARGET_WEBSITE,
                'extracted_at': datetime.now().isoformat(),
                'total_results': len(data),
                'results': data
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Raw data backup saved to JSON: {filepath}")
            return True

        except Exception as e:
            print(f"❌ Error saving JSON: {e}")
            return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def validate_config():
    """Validate configuration before running"""
    errors = []

    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE" or not GOOGLE_API_KEY:
        errors.append("❌ Please set your GOOGLE_API_KEY in the configuration section")

    if SEARCH_ENGINE_ID == "YOUR_SEARCH_ENGINE_ID_HERE" or not SEARCH_ENGINE_ID:
        errors.append("❌ Please set your SEARCH_ENGINE_ID in the configuration section")

    if not TARGET_WEBSITE or TARGET_WEBSITE == "mg-united.eu":
        errors.append("❌ Please set your TARGET_WEBSITE in the configuration section")

    if errors:
        print("🚫 Configuration Errors:")
        for error in errors:
            print(f"   {error}")
        print("\n📝 Please update the configuration section at the top of this file")
        return False

    return True

def main():
    """Main execution function"""
    print("🚀 Google Website Snippet Analyzer")
    print("=" * 60)

    # Validate configuration
    if not validate_config():
        sys.exit(1)

    # Display configuration
    print("⚙️  Configuration:")
    print(f"   Website: {TARGET_WEBSITE}")
    print(f"   Max Results: {MAX_RESULTS_PER_SEARCH}")
    print(f"   Output CSV: {OUTPUT_CSV_PATH}")
    print(f"   Delay: {DELAY_BETWEEN_REQUESTS}s between requests")
    print("=" * 60)

    # Initialize analyzer
    try:
        analyzer = GoogleWebsiteSnippetAnalyzer(GOOGLE_API_KEY, SEARCH_ENGINE_ID)
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        sys.exit(1)

    # Analyze website
    try:
        results = analyzer.analyze_website(TARGET_WEBSITE, MAX_RESULTS_PER_SEARCH)

        if not results:
            print("❌ No results found. Please check:")
            print("   - Your API credentials are correct")
            print("   - The website is indexed by Google")
            print("   - You haven't exceeded your API quota")
            sys.exit(1)

        # Save results
        csv_success = analyzer.save_to_csv(results, OUTPUT_CSV_PATH)
        json_success = analyzer.save_to_json(results, OUTPUT_JSON_PATH)

        if csv_success:
            print(f"✅ Analysis complete! Check your CSV file: {OUTPUT_CSV_PATH}")

        # Display summary statistics
        print("\n📈 Analysis Summary:")
        print(f"   Total pages analyzed: {len(results)}")

        snippet_lengths = [r['snippet_length'] for r in results]
        if snippet_lengths:
            print(f"   Average snippet length: {sum(snippet_lengths) / len(snippet_lengths):.1f} characters")
            print(f"   Snippet length range: {min(snippet_lengths)} - {max(snippet_lengths)} characters")

        structured_data_count = sum(1 for r in results if r.get('has_structured_data') == 'Yes')
        print(f"   Pages with structured data: {structured_data_count}/{len(results)} ({structured_data_count/len(results)*100:.1f}%)")

        print("\n🎉 Analysis completed successfully!")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
