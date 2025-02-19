import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import json
import logging
import random
import signal
import sys
from datetime import datetime, timedelta

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


class GracefulInterruptHandler:
    def __init__(self, timeout_minutes=290):
        self.interrupt_received = False
        self.timeout_received = False
        self.start_time = datetime.now()
        self.timeout_minutes = timeout_minutes
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        logging.info(
            "Received interrupt signal, will save progress and exit...")
        self.interrupt_received = True

    def should_stop(self):
        # Check for timeout
        if datetime.now() - self.start_time > timedelta(minutes=self.timeout_minutes):
            if not self.timeout_received:
                logging.info(
                    f"Reached {self.timeout_minutes} minute timeout, will save progress and exit...")
                self.timeout_received = True
            return True
        return self.interrupt_received or self.timeout_received


class ProgressManager:
    def __init__(self, checkpoint_file: str = "scraper_progress.json"):
        self.checkpoint_file = Path(checkpoint_file)
        self.temp_file = self.checkpoint_file.with_suffix('.tmp')
        self.backup_file = self.checkpoint_file.with_suffix('.bak')
        self.scraped_urls = set()
        self.all_player_data = []
        self.load_progress()

    def load_progress(self):
        # Try loading from main file, then temp, then backup
        for file in [self.checkpoint_file, self.temp_file, self.backup_file]:
            try:
                if file.exists():
                    with file.open('r') as f:
                        data = json.load(f)
                        self.scraped_urls = set(data['scraped_urls'])
                        self.all_player_data = data['player_data']
                        logging.info(
                            f"Loaded progress from {file}: {len(self.scraped_urls)} URLs scraped")
                        return
            except Exception as e:
                logging.warning(f"Error loading from {file}: {e}")
                continue

    def save_progress(self, final=False):
        try:
            # First save to temporary file
            with self.temp_file.open('w') as f:
                json.dump({
                    'scraped_urls': list(self.scraped_urls),
                    'player_data': self.all_player_data
                }, f)

            # If that succeeds, make a backup of current file if it exists
            if self.checkpoint_file.exists():
                self.checkpoint_file.replace(self.backup_file)

            # Then move temp file to main file
            self.temp_file.replace(self.checkpoint_file)

            if final:
                # Clean up backup and temp files
                self.backup_file.unlink(missing_ok=True)
                self.temp_file.unlink(missing_ok=True)

            logging.info(
                f"Progress saved: {len(self.scraped_urls)} URLs scraped, {len(self.all_player_data)} players found")

        except Exception as e:
            logging.error(f"Error saving progress: {e}")


def get_player_ids_and_urls(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    ids = []
    new_urls = set()
    career_table = soup.find('table', id=lambda x: x and 'career_totals' in x)
    if career_table:
        for link in career_table.find_all('a'):
            href = link.get('href', '')
            if '/players/' in href:
                player_id = href.split('/players/')[-1].strip()
                if player_id:
                    ids.append(player_id)
                    new_urls.add(f"https://stats.ncaa.com/players/{player_id}")
    return list(set(ids)), new_urls


def process_players(urls: list, timeout_minutes: int = 290) -> pd.DataFrame:
    progress = ProgressManager()
    interrupt_handler = GracefulInterruptHandler(timeout_minutes)
    to_scrape = set(urls) - progress.scraped_urls
    session = requests.Session()
    session.headers.update(headers)

    try:
        with tqdm(total=len(to_scrape)) as pbar:
            while to_scrape and not interrupt_handler.should_stop():
                current_batch = list(to_scrape)[:10]
                to_scrape = to_scrape - set(current_batch)

                for url in current_batch:
                    if interrupt_handler.should_stop():
                        break

                    if url in progress.scraped_urls:
                        pbar.update(1)
                        continue

                    try:
                        time.sleep(1 + random.uniform(0, 0.5))
                        response = session.get(
                            url, headers=headers, timeout=30)

                        if response.status_code == 430:
                            logging.warning(f"Rate limit hit at URL: {url}")
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

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        # Save final progress and clean up temporary files
        progress.save_progress(final=True)

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
            logging.error(f"Error reading {file}: {e}")
            continue

    if not dfs:
        raise ValueError("No valid roster files found")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates()
    logging.info(
        f"Combined {len(all_files)} files into DataFrame with {len(combined_df)} rows")
    return combined_df


def main(data_dir: str) -> None:
    data_dir = Path(data_dir)
    output_dir = data_dir / "rosters"
    output_dir.mkdir(exist_ok=True)

    try:
        df = combine_roster_files(output_dir)
        unique_links = df.player_url.unique()

        result = process_players(unique_links)

        # Save results with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f'master_{timestamp}.csv'
        result.to_csv(output_file, index=False)

        # Update master.csv
        master_file = output_dir / 'master.csv'
        if master_file.exists():
            existing_df = pd.read_csv(master_file)
            result = pd.concat([existing_df, result]).drop_duplicates()
        result.to_csv(master_file, index=False)

        logging.info(f"Results saved to {output_file} and master.csv")

        # Return success status
        return 0

    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        return 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    args = parser.parse_args()

    sys.exit(main(args.data_dir))
