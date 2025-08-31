import os
import pickle
import pandas as pd

def load_series_dfs(data_dir=None, filename='series_dfs.pkl'):
    """
    Lädt das Dictionary `series_dfs` aus einer Pickle-Datei im data-Ordner.

    Parameters
    ----------
    data_dir : str, optional
        Pfad zum data-Ordner. Standard: eine Ebene über dem aktuellen Arbeitsverzeichnis im Unterordner 'data'.
    filename : str, optional
        Dateiname der Pickle-Datei (Standard: 'series_dfs.pkl').

    Returns
    -------
    dict
        Das geladene Dictionary {series_id: DataFrame}.

    Raises
    ------
    FileNotFoundError
        Wenn die Datei nicht gefunden wird.
    """
    # Standard-Pfad: ../data
    if data_dir is None:
        # aktuelles Arbeitsverzeichnis
        cwd = os.getcwd()
        # gehe eine Ebene hoch und dann in 'data'
        data_dir = os.path.abspath(os.path.join(cwd, os.pardir, 'data'))
    file_path = os.path.join(data_dir, filename)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

    with open(file_path, 'rb') as f:
        series_dfs = pickle.load(f)

    return series_dfs

