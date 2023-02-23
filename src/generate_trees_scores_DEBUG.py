import numpy as np
import pandas as pd

from scipy import stats

def make_first_branches(det_t):
    new_branches = []
    for i, row in det_t.iterrows():
        new_br = np.array([row["id"]])
        new_branches.append(new_br)

    new_scores = [1]
    return new_branches, new_scores

def two_tail_prob(z):
    return 2*(1-stats.norm.cdf(z))



# TODO: deal with multiple detections at once properly
def make_new_branches(branches, scores, det_t, it, probz):

    new_branches = []
    new_scores = []

    old_it = len(branches[0])

    nan_append = np.empty(it - old_it) + np.NaN

    bz = []
    for br in branches:
        bz.append(np.append(br, nan_append))

    branches = bz
    # new_branches = []
    if len(det_t) == 1:
        # create new vector (case where this detection is a new individual)
        d = det_t.iloc[0]
        new_br = np.empty(it + 1)
        new_br[:] = np.NaN
        new_br[it] = d["id"]
        new_branches.append(new_br)

        new_score = 1
        new_scores.append(new_score)

        for i, br in enumerate(branches):
            # case where this is a repeat of the last observed
            new_br_repeat = np.append(br, d["id"])
            new_branches.append(new_br_repeat)

            # get the id of the previous detection in the current branch
            last_detection = np.array(br)[np.logical_not(np.isnan(br))][-1]
            scr = probz[int(last_detection), int(d["id"])]
            new_scores.append(scores[i] + scr)

            # case where it's not
            new_br_no_repeat = np.append(br, np.NaN)
            new_branches.append(new_br_no_repeat)
            new_scores.append(scores[i] + (1 - scr))

    else:
        for i, br in enumerate(branches):

            for i, d in det_t.iterrows():
                # create new vector (case where this detection is a new individual)
                new_br = np.empty(it + 1)
                new_br[:] = np.NaN
                new_br[it] = d["id"]

                new_branches.append(new_br)

                new_score = 1
                new_scores.append(new_score)


                new_br_repeat = np.append(br, d["id"])
                new_branches.append(new_br_repeat)

                # get the id of the previous detection in the current branch
                last_detection = np.array(br)[np.logical_not(np.isnan(br))][-1]
                scr = probz[int(last_detection), int(d["id"])]
                new_scores.append(scores[i] + scr)

            new_br_no_repeat = np.append(br, np.NaN)
            new_branches.append(new_br_no_repeat)
            new_scores.append(scores[i] + (1 - scr))
    return new_branches, new_scores


if __name__ == "__main__":

    faunastep = 0.5
    detections = pd.DataFrame()

    detections["id"] = [0, 1, 2]
    detections["time"] = [0, 1,2]

    # detections["time_chunk"] = [0,1,1]
    detections["object_location"] = [0, 0.5, 2]

    # detections = pd.read_csv('../data/detections.csv')

    # branches = []

    times = detections["time"].unique()


    n_dets = len(detections)
    dists = np.zeros((len(detections), len(detections)))
    Dtime = np.zeros((len(detections), len(detections)))
    for i in range(n_dets):
        for j in range(n_dets):
            dists[i, j] = np.abs(detections.loc[i]["object_location"] - detections.loc[j]["object_location"])
            Dtime[i, j] = np.abs(detections.loc[i]["time"] - detections.loc[j]["time"])

    z_stat = np.divide(dists, faunastep * Dtime)
    probz = two_tail_prob(z_stat)


    for it, t in enumerate(times):
        # get all detections made at that timestep
        det_t = detections[detections["time"] == t]

        if it == 0:
            branches, scores = make_first_branches(det_t)
        else:
            branches, scores = make_new_branches(branches, scores, det_t, it, probz)




    for i, br in enumerate(branches):
        print(br, scores[i])
        # print(scores[i])
    # print(branches)
