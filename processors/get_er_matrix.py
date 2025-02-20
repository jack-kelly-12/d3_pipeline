import pandas as pd
import numpy as np
from pathlib import Path


def get_expected_runs_matrix_2(base_cd, outs, runs_rest_of_inn):
    # Create initial dataframe and drop NA values all at once
    ER = pd.DataFrame({
        'base_cd': base_cd,
        'outs': outs,
        'runs_rest_of_inn': runs_rest_of_inn
    }).dropna()

    # Calculate ERV (no need for between filtering since we handle this in matrix creation)
    ER = (ER.groupby(['base_cd', 'outs'])
          .agg(
              ERV=('runs_rest_of_inn', 'mean'),
              count=('runs_rest_of_inn', 'size')
    )
        .reset_index())

    ER['ERV'] = ER['ERV'].round(3)

    # Initialize matrix with zeros
    ER_matrix = np.zeros((8, 3))

    # Fill matrix (adding 0 to base_cd and outs since R is 1-based)
    for _, row in ER.iterrows():
        base_idx = int(row['base_cd'])
        out_idx = int(row['outs'])
        if 0 <= base_idx < 8 and 0 <= out_idx < 3:  # Ensure indices are valid
            ER_matrix[base_idx, out_idx] = row['ERV']

    # Convert to DataFrame with proper labels
    ER_matrix = pd.DataFrame(
        ER_matrix,
        index=['_ _ _', '1B _ _', '_ 2B _', '1B 2B _',
               '_ _ 3B', '1B _ 3B', '_ 2B 3B', '1B 2B 3B'],
        columns=['0', '1', '2']
    )

    return ER_matrix


def main(data_dir, year):
    data_dir = Path(data_dir)
    divisions = range(1, 4)
    all_matrices = {}

    misc_dir = data_dir / 'miscellaneous'
    misc_dir.mkdir(exist_ok=True)

    for division in divisions:
        try:
            pbp_file = data_dir / 'play_by_play' / \
                f'd{division}_parsed_pbp_{year}.csv'

            if not pbp_file.exists():
                print(f"PBP file not found: {pbp_file}")
                continue

            pbp_df = pd.read_csv(pbp_file)
            print(f"Loaded {len(pbp_df)} rows for D{division} {year}")

            base_cd = pbp_df['base_cd_before']
            outs = pbp_df['outs_before']
            runs_rest_of_inn = pbp_df['runs_roi']

            matrix = get_expected_runs_matrix_2(
                base_cd, outs, runs_rest_of_inn)
            all_matrices[f"D{division}_{year}"] = matrix
            print(f"Processed D{division} {year}")

        except Exception as e:
            print(f"Error processing D{division} {year}: {str(e)}")
            continue

    final_dfs = []
    for name, matrix in all_matrices.items():
        division = int(name[1])
        year = int(name.split('_')[1])

        df = pd.DataFrame({
            'Division': division,
            'Year': year,
            'Bases': matrix.index,
            '0': matrix['0'],
            '1': matrix['1'],
            '2': matrix['2']
        })
        final_dfs.append(df)

    if not final_dfs:
        print("No matrices were generated!")
        return None

    final_df = pd.concat(final_dfs, ignore_index=True)
    final_df = final_df.sort_values(['Division', 'Year', 'Bases'])

    for division in divisions:
        output_file = misc_dir / f'd{division}_expected_runs_{year}.csv'
        division_df = final_df[final_df['Division'] == division]
        division_df.to_csv(
            output_file, index=False)
        print(f"Saved expected runs matrix for D{division} to {output_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    parser.add_argument('--year', required=True)
    args = parser.parse_args()

    main(args.data_dir, args.year)
