import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_last_detection_id


class track:
    def __init__(self, root):
        self.root = root
    def update(self, node):
        
        

class trackTrees:
    def __init__(self, detections):
        self.detections = detections

    def get_first_timestep(self):
        return self.detections.times.unique()[0]

    def make_root_nodes(self):
        t = self.get_first_timestep()
        dets = self.detections[self.detections.time == 0]

        for i, d in dets.iterrows():



    def update_tree(self, tree, dets):
        


def update_hyps(det, hyps):
    new_hyps = []

    detection_id = det.iloc[0].id

    # go through hypotheses
    for hyp_i, hyp in enumerate(hyps):


        # scenario where the new detection is a new individual
        # pad existing branches

        new_scen = [h + [np.nan] for h in hyp] + [list(np.empty(len(hyp[0])) * np.nan) + [detection_id]]
        new_hyps.append(new_scen)


        # go through the branches in the hypothesis
        for i, branch in enumerate(hyp):
            # calculate score
            last_detection = get_last_detection_id(branch)
            repeat_scen = [branch + [detection_id]]
            repeat_scen += [hyp[j] + [np.nan] for j in range(len(hyp)) if j != i]

            new_hyps.append(repeat_scen)


    return new_hyps



if __name__ == "__main__":
    var = 0.01

    detections = pd.DataFrame(columns=["id", "time", "x"])
    detections.id = [0,1,2]
    detections.time = [0,1,2]
    detections.x = [0, 1, 1.01]

    # fig, ax = plt.subplots()
    # ax.scatter(detections.x, detections.time)
    # plt.show()

    # t=0
    det = detections[detections.time == 0]
    hyps = [[[0]]]

    # t=1
    i=1
    det = detections[detections.time == i]
    hyps = update_hyps(det, hyps)


    a = 10
