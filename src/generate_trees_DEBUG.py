import numpy as np
import pandas as pd


def make_first_branches(det_t):
    new_branches = []
    for i, row in det_t.iterrows():
        new_br = np.array([row["id"]])
        new_branches.append(new_br)
    return new_branches

def make_new_branches(branches, det_t, it):

    new_branches = []

    old_it = len(branches[0])

    nan_append = np.empty(it - old_it) + np.NaN

    bz = []
    for br in branches:
        bz.append(np.append(br, nan_append))

    branches = bz
    # new_branches = []
    if len(det_t) == 1:
        d = det_t.iloc[0]
        new_br = np.empty(it + 1)
        new_br[:] = np.NaN
        new_br[it] = d["id"]
        new_branches.append(new_br)

        for i, br in enumerate(branches):
            new_br_repeat = np.append(br, d["id"])
            new_branches.append(new_br_repeat)

            new_br_no_repeat = np.append(br, np.NaN)
            new_branches.append(new_br_no_repeat)

    else:
        for i, br in enumerate(branches):
            new_br_no_repeat = np.append(br, np.NaN)
            new_branches.append(new_br_no_repeat)
            for i, d in det_t.iterrows():
                new_br = np.empty(it + 1)
                new_br[:] = np.NaN
                new_br[it] = d["id"]

                new_branches.append(new_br)

                new_br_repeat = np.append(br, d["id"])
                new_branches.append(new_br_repeat)

    return new_branches


if __name__ == "__main__":
    detections = pd.DataFrame()

    detections["id"] = [0, 1, 2]
    detections["time"] = [0, 1,2]

    # detections["time_chunk"] = [0,1,1]
    detections["object_location"] = [0, 0.5, 2]

    # detections = pd.read_csv('../data/detections.csv')

    # branches = []

    times = detections["time"].unique()

    for it, t in enumerate(times):
        # get all detections made at that timestep
        det_t = detections[detections["time"] == t]

        if it == 0:
            branches = make_first_branches(det_t)
        else:
            branches = make_new_branches(branches, det_t, it)




    for br in branches:
        print(br)
    # print(branches)
