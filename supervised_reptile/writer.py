"""Writes the given metrics in a csv."""
import numpy as np
import os
import sys
import pandas as pd


COLUMN_NAMES = ['Round', 'Average Test Acc']


def print_metrics(
        round_number,
        accuracy,
        metrics_dir,
        metrics_name):
    """Prints or appends the given metrics in a csv.

    Args:
        round_number: Number of the round the metrics correspond to. If
            0, then the file in path is overwritten. If not 0, we append to
            that file.
        client_ids: Ids of the clients. Not all ids must be in the following
            dicts.
        metrics: Dict keyed by client id. Each element is a dict of metrics
            for that client in the specified round. The dicts for all clients
            are expected to have the same set of keys.
        hierarchies: Dict keyed by client id. Each element is a list of hierarchies
            to which the client belongs.
        num_samples: Dict keyed by client id. Each element is the number of test
            samples for the client.
    """
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, '{}.csv'.format(metrics_name))

    columns = COLUMN_NAMES
    round_data = pd.DataFrame(columns=columns)
    current_round = {
            'Round': round_number,
            'Average Test Acc': accuracy
        }


    round_data.loc[len(round_data)] = current_round

    mode = 'w' if round_number == 0 else 'a'
    print_dataframe(round_data, path, mode)

def print_dataframe(df, path, mode='w'):
    """Writes the given dataframe in path as a csv"""
    header = mode == 'w'
    df.to_csv(path, mode=mode, header=header, index=False)
