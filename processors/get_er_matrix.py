import pandas as pd
import numpy as np


def get_expected_runs_matrix_2(base_cd, outs, runs_rest_of_inn):
    ER = pd.DataFrame({
        'base_cd': base_cd,
        'outs': outs,
        'runs_rest_of_inn': runs_rest_of_inn
    }).dropna()

    ER = ER[
        (ER['base_cd'].between(0, 7)) &
        (ER['outs'].between(0, 2))
    ]

    ER = (ER.groupby(['base_cd', 'outs'])
          .agg({
              'runs_rest_of_inn': ['mean', 'size']
          })
          .reset_index())

    ER.columns = ['base_cd', 'outs', 'ERV', 'count']

    ER['ERV'] = ER['ERV'].round(3)
    ER['state'] = ER['base_cd'].astype(str) + ' ' + ER['outs'].astype(str)

    ER_matrix = np.zeros((8, 3))
    for i in range(len(ER)):
        row = int(ER['base_cd'].iloc[i])
        col = int(ER['outs'].iloc[i])
        ER_matrix[row, col] = ER['ERV'].iloc[i]

    ER_matrix = pd.DataFrame(
        ER_matrix,
        index=['_ _ _', '1B _ _', '_ 2B _', '1B 2B _',
               '_ _ 3B', '1B _ 3B', '_ 2B 3B', '1B 2B 3B'],
        columns=['0', '1', '2']
    )

    return ER_matrix


def main(data_dir):
    year = 2025
    divisions = range(1, 4)
    all_matrices = {}

    for division in divisions:
        try:
            pbp_file = f'{data_dir}/play_by_play/d{division}_parsed_pbp_{year}.csv'

            if not pbp_file.exists():
                print(f"No PBP data for D{division} {year}")
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

    print(f"Saved {len(final_df)} rows of expected runs matrices")

    final_df.to_csv(
        f'{data_dir}/miscellaneous/d{division}_expected_runs_{year}.csv')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()

    main(args.data_dir)
