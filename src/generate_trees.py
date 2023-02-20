import igraph
import numpy as np
import itertools
import pandas as pd

from scipy import stats

from time import time


# TODO: figure out why it inf loops on multiple detections per timestep
# TODO: make sure padding is correct (or get rid of if unneccessary

def two_tail_prob(z):
    return 2*(1-stats.norm.cdf(z))


def calc_score(t0, t1, x0, x1, var):
    distance = np.abs(x1 - x0)
    dt = np.abs(t1 - t0)

    z = np.divide(distance, var * dt)

    score = two_tail_prob(z)

    return score


def calc_score_row(current_row, prev_row, var):
    t0 = prev_row["time"]
    t1 = current_row["time"]

    x0 = prev_row["object_location"]
    x1 = current_row["object_location"]

    score = calc_score(t0, t1, x0, x1, var)
    return score

# def make_hypothesis_tree(detections, faunastep):
#     branches = []
#     scores = []
#
#     times = detections.time.unique()
#     times.sort()
#
#     # go through each timestep
#     for t in times:
#         print("timestep: {}".format(t))
#
#         # get all detections made at that timestep
#         det_t = detections[detections["time"] == t]
#
#         # make some lists to fill
#         new_branches = []
#         new_scores = []
#
#         # iterate over the detections
#         for i, d in det_t.iterrows():
#             print(d["id"])
#             if t == 0:
#                 new_br = np.array([d["id"]])
#                 new_score = 1
#
#                 new_branches.append(new_br)
#                 new_scores.append(new_score)
#
#             else:
#                 tic = time()
#                 # create new vector (case where this detection is a new individual)
#                 new_br = np.empty(t + 1)
#                 new_br[:] = np.NaN
#                 new_br[t] = d["id"]
#
#                 print("Create new vector:{}s".format(time()-tic))
#
#                 # add this to the new branches for this timestep
#                 new_branches.append(new_br)
#
#                 # add a score (of zero) for this case
#                 new_scores.append(1)
#
#                 # iterate over existing branches
#                 for i, br in enumerate(branches):
#                     # get the id of the last detection in the current branch
#                     last_detection = np.array(br)[np.logical_not(np.isnan(br))][-1]
#
#                     # case where this is a repeat of the last observed
#                     new_br_repeat = np.append(br, d["id"])
#                     new_branches.append(new_br_repeat)
#
#                     # calculate score
#                     scr = calc_score_row(detections[detections["id"] == last_detection], d, faunastep)
#
#                     #                 print(len(scores), len(new_scores))
#                     new_scores.append(scores[i] + scr)
#
#                     # case where it's not
#                     new_br_no_repeat = np.append(br, np.NaN)
#                     new_branches.append(new_br_no_repeat)
#
#                     new_scores.append(scores[i] + 1 - scr)
#
#             branches = new_branches
#             scores = new_scores
#     return branches, scores


def make_hypothesis_tree(detections, faunastep):
    branches = []
    scores = []

    times = detections.time.unique()
    times.sort()

    n_dets = len(detections)

    dists = np.zeros((len(detections), len(detections)))
    Dtime = np.zeros((len(detections), len(detections)))
    for i in range(n_dets):
        for j in range(n_dets):
            dists[i, j] = np.abs(detections.loc[i]["object_location"] - detections.loc[j]["object_location"])
            Dtime[i, j] = np.abs(detections.loc[i]["time"] - detections.loc[j]["time"])

    z_stat = np.divide(dists, faunastep * Dtime)
    probz = two_tail_prob(z_stat)

    # go through each timestep
    for t in times:
        print("timestep: {}".format(t))

        # get all detections made at that timestep
        det_t = detections[detections["time"] == t]

        # make some lists to fill
        new_branches = []
        new_scores = []

        # iterate over the detections
        for i, d in det_t.iterrows():
            #             print(d["id"])
            if t == 0:
                new_br = np.array([d["id"]])
                new_score = 1

                new_branches.append(new_br)
                new_scores.append(new_score)

            else:
                # create new vector (case where this detection is a new individual)
                new_br = np.empty(t + 1)
                new_br[:] = np.NaN
                new_br[t] = d["id"]

                # add this to the new branches for this timestep
                new_branches.append(new_br)

                # add a score (of zero) for this case
                new_scores.append(1)

                print(len(branches))
                # iterate over existing branches
                for i, br in enumerate(branches):
                    print(i)
                    # get the id of the last detection in the current branch

                    last_detection = np.array(br)[np.logical_not(np.isnan(br))][-1]


                    # case where this is a repeat of the last observed
                    new_br_repeat = np.append(br, d["id"])
                    new_branches.append(new_br_repeat)

                    # calculate score
                    #                     scr = calc_score_row(detections[detections["id"] == last_detection], d, faunastep)

                    scr = probz[int(last_detection), int(d["id"])]

                    #                 print(len(scores), len(new_scores))
                    new_scores.append(scores[i] + scr)

                    # case where it's not
                    new_br_no_repeat = np.append(br, np.NaN)
                    new_branches.append(new_br_no_repeat)

                    new_scores.append(scores[i] + 1 - scr)

            branches = new_branches
            scores = new_scores
    return branches, scores


if __name__ == "__main__":
    detections = pd.DataFrame()

    detections["id"] = [0, 1, 2]
    detections["time"] = [0, 1,1]
    # detections["time_chunk"] = [0,1,1]
    detections["object_location"] = [0, 0.5, 2]


    branches, scores = make_hypothesis_tree(detections, 0.75)

    a=10