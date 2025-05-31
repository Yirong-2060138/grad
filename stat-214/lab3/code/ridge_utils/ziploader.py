# ridge_utils/ziploader.py

import zipfile
from io import TextIOWrapper, BytesIO
import numpy as np
from ridge_utils.textgrid import TextGrid
from ridge_utils.stimulus_utils import TRFile

class ZipDataLoader:
    def __init__(self, zip_path):
        self.zipf = zipfile.ZipFile(zip_path, "r")

    def list_files(self):
        return self.zipf.namelist()

    def load_textgrid(self, story):
        tg_path = f"ds003020/derivative/TextGrids/{story}.TextGrid"
        with self.zipf.open(tg_path) as f:
            return TextGrid(TextIOWrapper(f).read())

    def load_tr_file(self, story):
        tr_path = f"ds003020/derivative/func/{story}.tr"
        with self.zipf.open(tr_path) as f:
            lines = [line.strip() for line in TextIOWrapper(f)]
        trf = TRFile(None)
        for line in lines:
            time, label = line.split(" ", 1)
            time = float(time)
            if label in ("init-trigger", "trigger"):
                trf.trtimes.append(time)
            elif label == "sound-start":
                trf.soundstarttime = time
            elif label == "sound-stop":
                trf.soundstoptime = time
            else:
                trf.otherlabels.append((time, label))
        return trf

    def load_response(self, story):
        resp_path = f"ds003020/derivative/func/{story}_responses.npy"
        with self.zipf.open(resp_path) as f:
            return np.load(BytesIO(f.read()))
