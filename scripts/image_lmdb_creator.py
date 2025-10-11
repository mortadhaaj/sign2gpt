from argparse import ArgumentParser
import subprocess
import shutil
from pathlib import Path

import re
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from time import time
import io
import lmdb
import pickle
from glob import glob
from pqdm.processes import pqdm

parser = ArgumentParser(parents=[])

parser.add_argument(
    "--data_dir",
    type=str,
    default="dataset_creation/isharah_selfie",
)

parser.add_argument(
    "--lmdb_dir",
    type=str,
    default="isharah_selfie/lmdb_videos",
)

params, unknown = parser.parse_known_args()
data_dir = Path(params.data_dir)


ids = glob(f'{data_dir}/*.mp4') + glob(f'{data_dir}/*.mov')
print("Number of IDs =", len(ids))
ids = [x.rsplit('/',1)[-1] for x in ids]
# for id in tqdm(ids):
def do(id):
    id, ext = id.split(".")
    lmdb_dir = Path(params.lmdb_dir) / id
    video_path = str(data_dir / f"{id}.{ext}")

    n_bytes = 2**40

    tmp_dir = Path("/tmp") / f"TEMP_{time()}"
    env = lmdb.open(path=str(tmp_dir), map_size=n_bytes)
    txn = env.begin(write=True)

    cap = cv2.VideoCapture(video_path)

    if lmdb_dir.exists() and lmdb_dir.is_dir():
        exit()

    lmdb_dir.mkdir(parents=True, exist_ok=True)

    ind = 0
    counter = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        ret, frame = cap.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame).resize((256, 256))
        temp = io.BytesIO()
        img.save(temp, format="jpeg")
        temp.seek(0)
        txn.put(
            key=f"{ind}".encode("ascii"),
            value=temp.read(),
            dupdata=False,
        )
        ind += 1
        counter += 1

        if counter % 123 == 0 and counter != 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.put(
        key=f"details".encode("ascii"),
        value=pickle.dumps({"num_frames": ind, "id": id}, protocol=4),
        dupdata=False,
    )
    txn.commit()

    env.close()

    if lmdb_dir.exists():
        shutil.rmtree(lmdb_dir)
    shutil.move(f"{tmp_dir}", f"{lmdb_dir}")

r = pqdm(ids, do, n_jobs=40)
