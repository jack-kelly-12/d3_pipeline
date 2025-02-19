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
import signal
from contextlib import contextmanager
import sys

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


class TimeoutException(Exception):
    pass


@contextmanager
def timeout_handler(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timeout reached")

    # Register the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


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


def process_players(urls: List[str], timeout_minutes: int = 290) -> pd.DataFrame:
    progress = ProgressManager()
    to_scrape = set(urls) - progress.scraped_urls
    start_time = time.time()

    try:
        with timeout_handler(timeout_minutes * 60):  # Convert minutes to seconds
            with tqdm(total=len(to_scrape)) as pbar:
                while to_scrape:
                    current_batch = list(to_scrape)[:10]
                    to_scrape = to_scrape - set(current_batch)

                    for url in current_batch:
                        if url in progress.scraped_urls:
                            pbar.update(1)
                            continue

                        try:
                            time.sleep(1 + random.uniform(0, 0.5))
                            response = requests.get(url, headers=headers)

                            if response.status_code == 430:
                                logging.warning(
                                    f"Rate limit hit at URL: {url}")
                                progress.save_progress()
                                return pd.DataFrame(progress.all_player_data)

                            player_ids, new_urls = get_player_ids_and_urls(
                                response.content)
                            new_urls = new_urls - progress.scraped_urls
                            to_scrape.update(new_urls)
                            pbar.total = len(to_scrape) + \
                                len(progress.scraped_urls)

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

                            if len(progress.scraped_urls) % 10 == 0:
                                progress.save_progress()

                        except Exception as e:
                            logging.error(f"Error processing {url}: {e}")
                            continue

    except TimeoutException:
        logging.warning(
            f"Timeout reached after {timeout_minutes} minutes. Saving progress and exiting...")
        progress.save_progress()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        progress.save_progress()

    # Save final progress
    progress.save_progress()
    return pd.DataFrame(progress.all_player_data)


def main(data_dir: str) -> None:
    data_dir = Path(data_dir)
    output_dir = data_dir / "rosters"
    output_dir.mkdir(exist_ok=True)

    df = combine_roster_files(output_dir)
    unique_links = df.player_url.unique()

    # Process with 290-minute timeout (leaving buffer for GitHub Actions)
    result = process_players(unique_links, timeout_minutes=290)

    # Save results with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'master_{timestamp}.csv'
    result.to_csv(output_file, index=False)

    # Also update the main master.csv file
    master_file = output_dir / 'master.csv'
    if master_file.exists():
        existing_df = pd.read_csv(master_file)
        result = pd.concat([existing_df, result]).drop_duplicates()
    result.to_csv(master_file, index=False)

    logging.info(f"Results saved to {output_file} and master.csv")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    args = parser.parse_args()

    main(args.data_dir)
