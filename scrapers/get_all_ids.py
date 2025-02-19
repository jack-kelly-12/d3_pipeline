import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
from typing import List, Set, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}


def get_player_ids_and_urls(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    ids = []
    new_urls = set()
    career_table = soup.find('table', id=lambda x: x and 'career_totals' in x)
    if career_table:
        for link in career_table.find_all('a'):
            href = link['href']
            if '/players/' in href:
                player_id = href.split('/players/')[-1]
                ids.append(player_id)
                new_urls.add(f"https://stats.ncaa.com/players/{player_id}")
    return list(set(ids)), new_urls


def process_players(urls: List[str]) -> pd.DataFrame:
    scraped_urls = set()
    to_scrape = set(urls)
    all_player_data = []

    with tqdm(total=len(to_scrape)) as pbar:
        while to_scrape:
            current_batch = list(to_scrape)[:10]  # Process in batches of 10
            to_scrape = to_scrape - set(current_batch)

            for url in current_batch:
                if url in scraped_urls:
                    pbar.update(1)
                    continue

                try:
                    response = requests.get(url, headers=headers)
                    player_ids, new_urls = get_player_ids_and_urls(
                        response.content)

                    # Add new URLs to scrape
                    new_urls = new_urls - scraped_urls
                    to_scrape.update(new_urls)
                    pbar.total = len(to_scrape) + len(scraped_urls)

                    if player_ids:
                        min_id = min(player_ids)
                        unique_id = f'd3d-{min_id}'

                        for ncaa_id in player_ids:
                            all_player_data.append({
                                'ncaa_id': ncaa_id,
                                'unique_id': unique_id
                            })

                    scraped_urls.add(url)
                    pbar.update(1)
                    time.sleep(1)  # Respect rate limits

                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")
                    continue

    return pd.DataFrame(all_player_data)


def combine_roster_files(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    all_files = list(output_dir.glob("*_rosters_*.csv"))
    dfs = []

    for file in tqdm(all_files, desc="Reading roster files"):
        try:
            df = pd.read_csv(file)
            dfs.append(df)

        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    if not dfs:
        raise ValueError("No valid roster files found")

    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df = combined_df.drop_duplicates()

    print(
        f"Combined {len(all_files)} files into DataFrame with {len(combined_df)} rows")

    return combined_df


def main(data_dir: str) -> None:
    data_dir = Path(data_dir)
    output_dir = data_dir / "rosters"
    output_dir.mkdir(exist_ok=True)

    df = combine_roster_files(output_dir)
    unique_links = df.player_url.unique()

    result = process_players(unique_links)
    result.to_csv(output_dir / 'master.csv')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    args = parser.parse_args()

    main(args.data_dir)
