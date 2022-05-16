# P.S для своего спокойствия насчёт той части я делаю датасеты прямо как там т.е ctrl + c, ctrl + v
import os
import tarfile
import shutil
import hashlib
import glob
import random
import pickle
from datetime import datetime
import requests
dataset_path = 'dataset'
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)
def download_file(url: str, path: str):
    print('Downloading file ...')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    print('Download completed.')


def md5(path: str, chunk_size: int = 65536) -> str:
    hash_md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def untar(file_path: str, dest_path: str):
    print('Extracting file.')
    with tarfile.open(file_path, 'r:gz') as f:
        f.extractall(dest_path)
    print('Extraction completed.')



