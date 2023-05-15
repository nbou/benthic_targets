import numpy as np
import itertools
import igraph

from src.utils import generate_probability_matrix_2D, generate_probability_matrix, get_last_detection_id


# function to update hypotheses when only a single new detection is made
def update_hyps_scores_gate(det, hyps, scores, probz, bg_prob, score_thresh):
    new_hyps = []
    new_scores = []

    detection_id = det.iloc[0].id
    #     detection_time = det.iloc[0].time
    #     detection_pos = det.iloc[0].object_location

    # go through hypotheses
    for hyp_i, hyp in enumerate(hyps):
        existing_score = scores[hyp_i]

        # scenario where the new detection is a new individual
        # pad existing branches

        new_scen = [h + [np.nan] for h in hyp] + [list(np.empty(len(hyp[0])) * np.nan) + [detection_id]]
        new_hyps.append(new_scen)

        # score is the existing + background prob
        new_scores.append(existing_score + bg_prob)

        # go through the branches in the hypothesis
        for i, branch in enumerate(hyp):
            # calculate score
            last_detection = get_last_detection_id(branch)
            scre = probz[int(last_detection), int(detection_id)]

            # Do GATING, check if score is above threshold
            if scre > score_thresh:
                # scenario where the new detection is a repeat
                repeat_scen = [branch + [detection_id]]
                repeat_scen += [hyp[j] + [np.nan] for j in range(len(hyp)) if j != i]

                new_hyps.append(repeat_scen)

                new_scores.append(existing_score + scre)
            else:
                # print("Gated repeat detection {}, {}".format(last_detection, detection_id))
                continue

    return new_hyps, new_scores


# functions for simultaneous detections
def make_branches_multi(hypothesis, detection_ids, probz, score_thresh):
    # Create the case where all detections are new individuals
    new_scen = [h + [np.nan] for h in hypothesis] + [list(np.empty(len(hypothesis)) * np.nan) + [d] for d in
                                                     detection_ids]

    repeat_scen = []
    for h in hypothesis:
        for d in detection_ids:
            last_detection = get_last_detection_id(h)
            scre = probz[int(last_detection), int(d)]
            if scre > score_thresh:
                repeat_scen.append(h + [d])
            else:
                # print("Gated repeat detection {}, {}".format(last_detection, d))
                continue
    all_branches = new_scen + repeat_scen
    return all_branches


def score_new_branches(branches, bg_prob, probz):
    scores = [0] * len(branches)

    for i, br in enumerate(branches):
        # if it's just the padded branches form the existing hypothesis then score is zero
        if np.isnan(br[-1]):
            scores[i] = 0
        # if it ends in a new detections, but all nans otherwise then score is the background
        elif np.sum(np.logical_not(np.isnan(br[:-1]))) == 0:
            scores[i] = bg_prob
        # otherwise, calculate the repeat prob
        else:
            current_det = br[-1]
            last_det = get_last_detection_id(br[:-1])
            scores[i] = probz[int(last_det), int(current_det)]

    return scores


def make_tree_compat(trees):
    trees = np.array(trees)
    tree_compat = np.zeros((len(trees), len(trees)))

    for step in trees.transpose():
        unique, counts = np.unique(step, return_counts=True)
        reps = unique[np.where(counts > 1)]
        for r in reps:
            if r >= 0:
                whr = np.where(step == r)
                for pair in itertools.permutations(whr[0], r=2):
                    tree_compat[pair] = 1
    return tree_compat


def make_hypotheses_from_trees(trees):
    tree_compat = make_tree_compat(trees)
    g = igraph.Graph()
    nverts = len(trees)
    g.add_vertices(nverts)
    labels = []
    edges = []
    for i in range(nverts):
        labels.append(str(i))
        for ed in np.where(tree_compat[i] == 1)[0]:
            edges.append((i, ed))

    g.vs["label"] = labels

    g.add_edges(edges)
    hyps = g.maximal_independent_vertex_sets()
    return hyps


def update_hyps_scores_multi(dets, hyps, scores, probz, bg_prob, score_thresh):
    new_hyps = []
    new_scores = []

    detection_ids = dets["id"].to_list()

    for hyp_i, hyp in enumerate(hyps):
        existing_score = scores[hyp_i]

        # for h in hyp:
        branches = make_branches_multi(hyp, detection_ids, probz, score_thresh)
        br_scores = score_new_branches(branches, bg_prob, probz)

        hyp_inds = make_hypotheses_from_trees(branches)

        for hind in hyp_inds:
            new_hyps.append([branches[i] for i in hind])

            new_scores.append(existing_score + np.sum([br_scores[i] for i in hind]))

    return new_hyps, new_scores


def drop_low_scoring_hyps(hyps, scores, n_to_keep):
    idx = np.argpartition(scores, -n_to_keep)[-n_to_keep:]

    new_scores = [scores[i] for i in idx]
    new_hyps = [hyps[i] for i in idx]

    return new_hyps, new_scores

def drop_50(hyps, scores):
    n_to_keep = int(np.ceil(len(scores)/2))
    return drop_low_scoring_hyps(hyps, scores, n_to_keep)


def build_and_score_hypotheses_multi(detections, faunastep, bg_prob,  score_thresh, max_hyps=30000, d2=True):
    if d2:
        probz = generate_probability_matrix_2D(detections, faunastep)
    else:
        probz = generate_probability_matrix(detections, faunastep)
    # make hypotheses and scores for first timestep
    scores = []
    hyps = []

    t0 = detections.time.unique()[0]
    for i, row in detections[detections["time"] == t0].iterrows():
        hyps.append([[row["id"]]])
        scores.append(bg_prob)

    # update hypotheses and scores for the subsequent timesteps
    for t in detections.time.unique()[1:]:
        print("Processing detections from {}s, {} existing hypothesess".format(t, len(scores)))

        dets = detections[detections["time"] == t]

        if len(dets) == 1:

            hyps, scores = update_hyps_scores_gate(dets,
                                                   hyps,
                                                   scores,
                                                   probz,
                                                   bg_prob,
                                                   score_thresh)

        else:
            hyps, scores = update_hyps_scores_multi(dets,
                                                    hyps,
                                                    scores,
                                                    probz,
                                                    bg_prob,
                                                    score_thresh)

        if len(scores) > max_hyps:
            hyps, scores = drop_low_scoring_hyps(hyps, scores, n_to_keep=max_hyps)
    return hyps, scores

def build_and_score_hypotheses_multi_drop50(detections, faunastep, bg_prob,  score_thresh, d2=False):
    if d2:
        probz = generate_probability_matrix_2D(detections, faunastep)
    else:
        probz = generate_probability_matrix(detections, faunastep)
    # make hypotheses and scores for first timestep
    scores = []
    hyps = []

    t0 = detections.time.unique()[0]
    for i, row in detections[detections["time"] == t0].iterrows():
        hyps.append([[row["id"]]])
        scores.append(bg_prob)

    # update hypotheses and scores for the subsequent timesteps
    for t in detections.time.unique()[1:]:
        print("Processing detections from {}s, {} existing hypothesess".format(t, len(scores)))

        dets = detections[detections["time"] == t]

        if len(dets) == 1:

            hyps, scores = update_hyps_scores_gate(dets,
                                                   hyps,
                                                   scores,
                                                   probz,
                                                   bg_prob,
                                                   score_thresh)

        else:
            hyps, scores = update_hyps_scores_multi(dets,
                                                    hyps,
                                                    scores,
                                                    probz,
                                                    bg_prob,
                                                    score_thresh)

        if len(scores) > 10000:
            if len(scores) > 20000:
                hyps, scores = drop_low_scoring_hyps(hyps, scores, 20000)
            else:
                hyps, scores = drop_50(hyps, scores)
    return hyps, scores


if __name__ == "__main__":
    import pandas as pd
    from src.hypotheses import generate_probability_matrix_2D

    import matplotlib.pyplot as plt


    detections = pd.read_csv('../figs/detections.csv')
    faunastep = 0.1
    bg_prob = 10 / 25
    score_thrsh = 0.1

    detections.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

    # detections = detections.iloc[::3, :]
    hyps, scores = build_and_score_hypotheses_multi(detections, faunastep, bg_prob, score_thrsh, max_hyps=20000, d2=True)

    lens = [len(h) for h in hyps]

    fig, ax = plt.subplots(1,2)

    ax[0].scatter(lens, scores)
    ax[1].scatter(lens, np.divide(scores, lens))
    fig.show()

    a = 0
