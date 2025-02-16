import pandas as pd
import numpy as np
import argparse
from pathlib import Path


class LinearWeightsCalculator:
    EVENT_CODES = {
        'out': [2, 3, 6],
        'walk': [14],
        'hit_by_pitch': [16],
        'single': [20],
        'double': [21],
        'triple': [22],
        'home_run': [23]
    }

    def __init__(self, re24_matrix):
        self.re24_matrix = re24_matrix

    def get_re(self, base_cd, outs):
        if isinstance(base_cd, (pd.Series, np.ndarray)):
            base_cd = np.clip(base_cd + 1, 1, 8)
            outs = np.clip(outs + 1, 1, 3)
            return np.array([self.re24_matrix[b-1, o-1]
                             if not (np.isnan(b) or np.isnan(o)) else 0
                             for b, o in zip(base_cd, outs)])
        else:
            if np.isnan(base_cd) or np.isnan(outs):
                return 0
            base_cd = np.clip(base_cd + 1, 1, 8)
            outs = np.clip(outs + 1, 1, 3)
            return self.re24_matrix[base_cd-1, outs-1]

    def calculate_linear_weights(self, pbp_data):
        event_mapping = {}
        for event_name, codes in self.EVENT_CODES.items():
            for code in codes:
                event_mapping[code] = event_name

        events = np.array(pbp_data['event_cd'])
        bases_before = np.array(pbp_data['base_cd_before'])
        outs_before = np.array(pbp_data['outs_before'])
        runs_on_play = np.array(pbp_data['runs_on_play'])
        inn_end = np.array(pbp_data['inn_end'])

        re_start = self.get_re(bases_before, outs_before)

        re_end = np.zeros_like(re_start)
        re_end[:-1] = self.get_re(bases_before[1:], outs_before[1:])
        re_end[inn_end == 1] = 0

        re24 = re_end - re_start + runs_on_play

        event_counts = {event: 0 for event in self.EVENT_CODES.keys()}
        event_re24_sums = {event: 0 for event in self.EVENT_CODES.keys()}

        for event_code in np.unique(events):
            if event_code in event_mapping:
                event_name = event_mapping[event_code]
                mask = events == event_code
                event_counts[event_name] += np.sum(mask)
                event_re24_sums[event_name] += np.sum(re24[mask])

        linear_weights = pd.DataFrame({
            'events': list(event_counts.keys()),
            'count': list(event_counts.values()),
            'linear_weights_above_average': [
                round(event_re24_sums[event] / max(count, 1), 3)
                for event, count in event_counts.items()
            ]
        })

        out_weight = linear_weights.loc[
            linear_weights['events'] == 'out',
            'linear_weights_above_average'
        ].iloc[0]

        linear_weights['linear_weights_above_outs'] = (
            linear_weights['linear_weights_above_average'] - out_weight
        )

        return linear_weights.sort_values('linear_weights_above_average', ascending=False)

    def calculate_normalized_weights(self, linear_weights, stats):
        total_value = np.sum(
            linear_weights['linear_weights_above_outs'] *
            linear_weights['count']
        )
        total_pa = linear_weights['count'].sum()
        denominator = total_value / total_pa

        league_obp = (
            (stats['H'].sum() + stats['BB'].sum() + stats['HBP'].sum()) /
            (stats['AB'].sum() + stats['BB'].sum() + stats['HBP'].sum() +
             stats['SF'].sum() + stats['SH'].sum())
        )

        woba_scale = league_obp / denominator

        result = linear_weights.copy()
        result['normalized_weight'] = round(
            result['linear_weights_above_outs'] * woba_scale,
            3
        )

        woba_scale_row = pd.DataFrame([{
            'events': 'wOBA scale',
            'linear_weights_above_outs': np.nan,
            'count': np.nan,
            'normalized_weight': round(woba_scale, 3)
        }])

        return pd.concat([result, woba_scale_row], ignore_index=True)


def main(data_dir):
    year = 2025

    for division in range(1, 4):
        re24_matrix = pd.read_csv(data_dir /
                                  'miscellaneous' / 'd{division}_expected_runs_{year}.csv')
        calculator = LinearWeightsCalculator(re24_matrix)
        print(f'Processing Division {division}')

        pbp_file = data_dir / 'play_by_play' / \
            f'd{division}_parsed_pbp_{year}.csv'
        stats_file = data_dir / 'statistics' / \
            f'd{division}_batting_{year}.csv'

        if not pbp_file.exists() or not stats_file.exists():
            print(f"Missing required files for Division {division}")
            continue

        pbp_data = pd.read_csv(pbp_file)
        stats_data = pd.read_csv(stats_file)

        linear_weights = calculator.calculate_linear_weights(pbp_data)
        normalized_weights = calculator.calculate_normalized_weights(
            linear_weights, stats_data)

        output_file = data_dir / 'miscellaneous' / \
            f'd{division}_linear_weights_{year}.csv'
        normalized_weights.to_csv(output_file, index=False)
        print(f"Saved weights to {output_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()

    main(args.data_dir)
