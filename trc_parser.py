import numpy as np
import re

class TRC:
    def __init__(self, data):
        self.data = data
        self.frames = None
        self.n_frames = -1
        self.tokenize()

    def tokenize(self):
        preidx_frame = []
        for idx, string in enumerate(self.data):
            line = re.split('\\s+', string.strip())

            if idx > 4: # begin index of position data per frame
                preidx_frame.append(list(map(float, line[2:])))

        preidx_frame = np.array(preidx_frame).reshape(-1, 19, 3)

        # reindex
        # trc index -> 0: ThumbCMC ... 16: Backhand1
        # nn index -> 0: Backhand1 ... 3: ThumbCMC
        self.frames = np.zeros_like(preidx_frame)
        self.frames[:, :3, :] = preidx_frame[:, 16:, :]
        self.frames[:, 3:, :] = preidx_frame[:, :16, :]
        self.n_frames = self.frames.shape[0]
