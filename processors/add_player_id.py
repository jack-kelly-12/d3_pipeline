import pandas as pd
from typing import List, Tuple
from collections import defaultdict
from pathlib import Path


class PlayerMatcher:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.next_id = 1
        self.player_info = defaultdict(list)
        self.decisions = []
        self.used_ids = set()

    def _generate_id(self) -> str:
        while True:
            new_id = f"d3-playerid-{self.next_id}"
            self.next_id += 1
            if new_id not in self.used_ids:
                self.used_ids.add(new_id)
                return new_id

    def _clean_name(self, names: pd.Series) -> pd.Series:
        return names.str.strip().str.upper()

    def _is_valid_year_progression(self, years: List[int], yr_values: List[str]) -> bool:
        if len(years) <= 1:
            return True

        yr_map = {'FR': 1, 'SO': 2, 'JR': 3, 'SR': 4,
                  'Fr.': 1, 'So.': 2, 'Jr.': 3, 'Sr.': 4}
        numeric_years = [yr_map.get(yr.upper() if isinstance(
            yr, str) else yr, 0) for yr in yr_values]

        year_pairs = list(zip(sorted(years), sorted(numeric_years)))
        for i in range(len(year_pairs)-1):
            curr_year, curr_yr = year_pairs[i]
            next_year, next_yr = year_pairs[i+1]

            if next_year != curr_year + 1:
                return False

            if next_yr < curr_yr:
                return False

        return True

    def process_files(self, batting_files: List[str], pitching_files: List[str], roster_files: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.reset_state()

        batting_dfs = []
        pitching_dfs = []
        roster_dfs = []

        for file in sorted(batting_files):
            df = pd.read_csv(file)
            year = int(file.split('_')[-1].split('.')[0])
            df['year'] = year
            df['data_type'] = 'batting'
            df['original_id'] = df.index
            batting_dfs.append(df)

        for file in sorted(pitching_files):
            df = pd.read_csv(file)
            year = int(file.split('_')[-1].split('.')[0])
            df['year'] = year
            df['data_type'] = 'pitching'
            df['original_id'] = df.index
            pitching_dfs.append(df)

        for file in sorted(roster_files):
            df = pd.read_csv(file)
            year = int(file.split('_')[-1].split('.')[0])
            df['year'] = year
            df['data_type'] = 'roster'
            df['original_id'] = df.index
            roster_dfs.append(df)

        batting_df = pd.concat(batting_dfs, ignore_index=True)
        pitching_df = pd.concat(pitching_dfs, ignore_index=True)
        roster_df = pd.concat(roster_dfs, ignore_index=True)

        batting_df['clean_name'] = self._clean_name(batting_df['player_name'])
        pitching_df['clean_name'] = self._clean_name(
            pitching_df['player_name'])
        roster_df['clean_name'] = self._clean_name(roster_df['player_name'])

        batting_df = batting_df.rename(columns={'Yr': 'class'})
        pitching_df = pitching_df.rename(columns={'Yr': 'class'})

        self._match_players(batting_df, pitching_df, roster_df)

        batting_output = self._create_output_dataframe(batting_df)
        pitching_output = self._create_output_dataframe(pitching_df)
        roster_output = self._create_output_dataframe(roster_df)

        return batting_output, pitching_output, roster_output

    def _match_players(self, batting_df: pd.DataFrame, pitching_df: pd.DataFrame, roster_df: pd.DataFrame):
        combined_df = pd.concat(
            [batting_df, pitching_df, roster_df], ignore_index=True)

        for (name, school), group in combined_df.groupby(['clean_name', 'team_name']):
            years = sorted(group['year'].unique())
            class_values = group['class'].unique()

            if self._is_valid_year_progression(years, class_values):
                self._assign_same_id(group, confidence='high')
                continue

            positions = group['Pos'].fillna(
                '').unique() if 'Pos' in group.columns else []
            positions = positions.tolist() + group['position'].fillna(
                '').unique().tolist() if 'position' in group.columns else positions

            if self._are_positions_compatible(positions):
                self._assign_same_id(group, confidence='medium')
            else:
                for _, subgroup in group.groupby('data_type'):
                    self._assign_new_id(subgroup, confidence='low')

    def _are_positions_compatible(self, positions: List[str]) -> bool:
        positions = [str(pos).upper()
                     for pos in positions if pd.notna(pos) and str(pos).strip()]

        if not positions:
            return True

        if 'P' in positions or 'PITCHER' in positions:
            return True

        position_groups = {
            'infield': {'IF', '1B', '2B', '3B', 'SS', 'INF'},
            'outfield': {'OF', 'LF', 'CF', 'RF', 'OUTFIELD'},
            'battery': {'C', 'P', 'CATCHER', 'PITCHER'},
            'utility': {'DH', 'UT', 'PR', 'PH', 'UTILITY'}
        }

        player_groups = set()
        for pos in positions:
            for group, valid_pos in position_groups.items():
                if pos in valid_pos:
                    player_groups.add(group)

        if 'utility' in player_groups:
            return True

        return len(player_groups - {'utility'}) <= 1

    def _assign_new_id(self, group: pd.DataFrame, confidence: str):
        internal_id = self._generate_id()
        for _, row in group.iterrows():
            self.player_info[internal_id].append(row.to_dict())
            self.decisions.append({
                'internal_id': internal_id,
                'original_id': row['original_id'],
                'decision': 'new_id',
                'confidence': confidence
            })

    def _assign_same_id(self, group: pd.DataFrame, confidence: str):
        internal_id = self._generate_id()
        for _, row in group.iterrows():
            self.player_info[internal_id].append(row.to_dict())
            self.decisions.append({
                'internal_id': internal_id,
                'original_id': row['original_id'],
                'decision': 'matched',
                'confidence': confidence
            })

    def _create_output_dataframe(self, df_type: pd.DataFrame) -> pd.DataFrame:
        mapped_records = []
        df_type = df_type.rename(
            columns={'player_id': 'ncaa_id'}) if 'player_id' in df_type.columns else df_type
        original_columns = df_type.columns.tolist()
        data_type = df_type['data_type'].iloc[0]

        for internal_id, seasons in self.player_info.items():
            for season in seasons:
                if season['data_type'] == data_type:
                    filtered_season = {
                        k: v for k, v in season.items() if k in original_columns}
                    filtered_season['player_id'] = internal_id
                    mapped_records.append(filtered_season)

        result_df = pd.DataFrame(mapped_records)

        columns_to_keep = [col for col in original_columns if col not in [
            'clean_name', 'original_id', 'data_type']] + ['player_id']
        return result_df[columns_to_keep]


def main(data_dir):
    data_dir = Path(data_dir)
    stats_dir = data_dir / 'stats'
    rosters_dir = data_dir / 'rosters'

    # Ensure directories exist
    stats_dir.mkdir(exist_ok=True)
    rosters_dir.mkdir(exist_ok=True)

    # Get file lists using pathlib
    batting_files = sorted(list(stats_dir.glob('d3_batting_2*.csv')))
    pitching_files = sorted(list(stats_dir.glob('d3_pitching_2*.csv')))
    roster_files = sorted(list(rosters_dir.glob('d3_rosters_2*.csv')))

    matcher = PlayerMatcher()
    mapped_batting, mapped_pitching, mapped_rosters = matcher.process_files(
        batting_files, pitching_files, roster_files)

    for year in range(2021, 2025):
        # Process batting data
        df_batting = mapped_batting.query(
            f'year == {year}').rename(columns={'class': 'Yr'})

        # Process pitching data
        df_pitching = mapped_pitching.query(
            f'year == {year}').rename(columns={'class': 'Yr'})

        # Process roster data
        df_roster = mapped_rosters.query(f'year == {year}')

        # Save processed files using pathlib
        batting_output = stats_dir / f'd3_batting_{year}.csv'
        pitching_output = stats_dir / f'd3_pitching_{year}.csv'
        roster_output = rosters_dir / f'd3_rosters_{year}.csv'

        df_batting.to_csv(batting_output, index=False)
        df_pitching.to_csv(pitching_output, index=False)
        df_roster.to_csv(roster_output, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    args = parser.parse_args()

    main(args.data_dir)
