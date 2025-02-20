import pandas as pd
from pathlib import Path
import json
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('id_mapper.log'),
        logging.StreamHandler()
    ]
)


def safe_map_id(player_id: str, id_mapping: dict) -> str:
    """
    Maps player_id using id_mapping, returning original id if no mapping exists.

    Args:
        player_id: The original player ID
        id_mapping: Dictionary of ID mappings

    Returns:
        Mapped ID if available, otherwise original ID
    """
    if pd.isna(player_id):
        return player_id
    str_id = str(player_id)
    return id_mapping.get(str_id, str_id)


def map_player_ids(data_dir: str | Path) -> None:
    """
    Maps unique_id from scraper progress to player_id in batting, pitching, and roster files.
    Keeps existing ID if no mapping is found.

    Args:
        data_dir: Path to the data directory containing stats/ and rosters/ subdirectories
    """
    data_dir = Path(data_dir)
    progress_file = Path("scraper_progress.json")

    # Load ID mapping from progress file
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
            id_mapping = {
                str(item['ncaa_id']): item['unique_id']
                for item in progress_data['player_data']
            }
    except Exception as e:
        logging.error(f"Error loading progress file: {e}")
        return

    # Process batting files
    batting_files = list((data_dir / "stats").glob("d3_batting_*.csv"))
    for file in tqdm(batting_files, desc="Processing batting files"):
        try:
            df = pd.read_csv(file, dtype={'player_id': str})
            if 'player_id' in df.columns:
                df['player_id'] = df['player_id'].apply(
                    lambda x: safe_map_id(x, id_mapping))
                df.to_csv(file, index=False)
                logging.info(f"Updated batting file: {file}")
        except Exception as e:
            logging.error(f"Error processing batting file {file}: {e}")

    # Process pitching files
    pitching_files = list((data_dir / "stats").glob("d3_pitching_*.csv"))
    for file in tqdm(pitching_files, desc="Processing pitching files"):
        try:
            df = pd.read_csv(file, dtype={'player_id': str})
            if 'player_id' in df.columns:
                df['player_id'] = df['player_id'].apply(
                    lambda x: safe_map_id(x, id_mapping))
                df.to_csv(file, index=False)
                logging.info(f"Updated pitching file: {file}")
        except Exception as e:
            logging.error(f"Error processing pitching file {file}: {e}")

    # Process roster files
    roster_files = list((data_dir / "rosters").glob("d3_rosters_*.csv"))
    for file in tqdm(roster_files, desc="Processing roster files"):
        try:
            df = pd.read_csv(file, dtype={'player_id': str})
            if 'player_id' in df.columns:
                df['player_id'] = df['player_id'].apply(
                    lambda x: safe_map_id(x, id_mapping))
                df.to_csv(file, index=False)
                logging.info(f"Updated roster file: {file}")
        except Exception as e:
            logging.error(f"Error processing roster file {file}: {e}")

    # Update master roster file if it exists
    master_file = data_dir / "rosters" / "master.csv"
    if master_file.exists():
        try:
            df = pd.read_csv(master_file, dtype={'player_id': str})
            if 'player_id' in df.columns:
                df['player_id'] = df['player_id'].apply(
                    lambda x: safe_map_id(x, id_mapping))
                df.to_csv(master_file, index=False)
                logging.info("Updated master roster file")
        except Exception as e:
            logging.error(f"Error processing master file: {e}")

    logging.info("Completed ID mapping")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    args = parser.parse_args()

    map_player_ids(args.data_dir)
