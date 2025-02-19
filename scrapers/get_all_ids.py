import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import json
from typing import List, Set, Dict, Tuple
import logging
import random

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


class ProgressManager:
    def __init__(self, checkpoint_file: str = "scraper_progress.json"):
        self.checkpoint_file = Path(checkpoint_file)
        self.scraped_urls = set()
        self.all_player_data = []
        self.load_progress()

    def load_progress(self):
        if self.checkpoint_file.exists():
            with self.checkpoint_file.open('r') as f:
                data = json.load(f)
                self.scraped_urls = set(data['scraped_urls'])
                self.all_player_data = data['player_data']
                logging.info(
                    f"Loaded progress: {len(self.scraped_urls)} URLs scraped, {len(self.all_player_data)} players found")

    def save_progress(self):
        with self.checkpoint_file.open('w') as f:
            json.dump({
                'scraped_urls': list(self.scraped_urls),
                'player_data': self.all_player_data
            }, f)
            logging.info(
                f"Progress saved: {len(self.scraped_urls)} URLs scraped, {len(self.all_player_data)} players found")


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
    progress = ProgressManager()
    to_scrape = set(urls) - progress.scraped_urls

    with tqdm(total=len(to_scrape)) as pbar:
        while to_scrape:
            current_batch = list(to_scrape)[:10]  # Process in batches of 10
            to_scrape = to_scrape - set(current_batch)

            for url in current_batch:
                if url in progress.scraped_urls:
                    pbar.update(1)
                    continue

                try:
                    # Add jitter to avoid synchronized requests
                    time.sleep(1 + random.uniform(0, 0.5))

                    response = requests.get(url, headers=headers)

                    if response.status_code == 430:
                        logging.warning(f"Rate limit hit at URL: {url}")
                        # Save progress before exiting
                        progress.save_progress()
                        # Return current progress as DataFrame
                        return pd.DataFrame(progress.all_player_data)

                    player_ids, new_urls = get_player_ids_and_urls(
                        response.content)

                    # Add new URLs to scrape
                    new_urls = new_urls - progress.scraped_urls
                    to_scrape.update(new_urls)
                    pbar.total = len(to_scrape) + len(progress.scraped_urls)

                    if player_ids:
                        min_id = min(player_ids)
                        unique_id = f'd3d-{min_id}'

                        for ncaa_id in player_ids:
                            progress.all_player_data.append({
                                'ncaa_id': ncaa_id,
                                'unique_id': unique_id
                            })

                    progress.scraped_urls.add(url)
                    pbar.update(1)

                    # Save progress every 10 URLs
                    if len(progress.scraped_urls) % 10 == 0:
                        progress.save_progress()

                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")
                    continue

    # Save final progress
    progress.save_progress()
    return pd.DataFrame(progress.all_player_data)


def combine_roster_files(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    all_files = list(output_dir.glob("d3_rosters_*.csv"))
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

    # Save results with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result.to_csv(output_dir / f'master_{timestamp}.csv')
    logging.info(f"Results saved to master_{timestamp}.csv")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    args = parser.parse_args()

    main(args.data_dir)
