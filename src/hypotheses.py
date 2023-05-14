import numpy as np
from scipy import stats
import igraph
import itertools


def two_tail_prob(z):
    return 2*(1-stats.norm.cdf(z))

def get_last_detection_id(br):
    return np.array(br)[np.logical_not(np.isnan(br))][-1]


def update_hyps_scores(det, hyps, scores, probz, bg_prob):
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
            # scenario where the new detection is a repeat
            repeat_scen = [branch + [detection_id]]
            repeat_scen += [hyp[j] + [np.nan] for j in range(len(hyp)) if j != i]

            new_hyps.append(repeat_scen)
            # calculate score
            last_detection = get_last_detection_id(branch)

            new_scores.append(existing_score + probz[int(last_detection), int(detection_id)])
    return new_hyps, new_scores


def generate_probability_matrix(detections, faunastep):

    n_dets = len(detections)
    dists = np.zeros((len(detections), len(detections)))
    Dtime = np.zeros((len(detections), len(detections)))
    for i in range(n_dets):
        for j in range(n_dets):
            dists[i, j] = np.abs(detections.loc[i]["object_location"] - detections.loc[j]["object_location"])
            Dtime[i, j] = np.abs(detections.loc[i]["time"] - detections.loc[j]["time"])

    z_stat = np.divide(dists, faunastep * Dtime)
    return two_tail_prob(z_stat)


def generate_probability_matrix_2D(detections, faunastep):

    n_dets = len(detections)
    dists = np.zeros((len(detections), len(detections)))
    Dtime = np.zeros((len(detections), len(detections)))
    for i in range(n_dets):
        for j in range(n_dets):
            dists[i, j] = np.linalg.norm(np.array((detections.loc[i]["x"], detections.loc[i]["y"])) -
                                         np.array((detections.loc[j]["x"], detections.loc[j]["y"])))
            Dtime[i, j] = np.abs(detections.loc[i]["time"] - detections.loc[j]["time"])

    z_stat = np.divide(dists, faunastep * Dtime)
    return two_tail_prob(z_stat)

def build_and_score_hypotheses(detections, faunastep, bg_prob):
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
        hyps, scores = update_hyps_scores(detections[detections["time"]==t],
                                          hyps,
                                          scores, probz,
                                          bg_prob)

    return hyps, scores


def build_and_score_hypotheses_multi(detections, faunastep, bg_prob, d2=False):

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

        dets = detections[detections["time"]==t]

        if len(dets) == 1:

            hyps, scores = update_hyps_scores(dets,
                                              hyps,
                                              scores,
                                              probz,
                                              bg_prob)

        else:
            hyps, scores = update_hyps_scores_multi(dets,
                                                    hyps,
                                                    scores,
                                                    probz,
                                                    bg_prob)

    return hyps, scores

# functions for simultaneous detections
def make_branches_multi(hypothesis, detection_ids):
    # Create the case where all detections are new individuals
    new_scen = [h + [np.nan] for h in hypothesis] + [list(np.empty(len(hypothesis)) * np.nan) + [d] for d in detection_ids]

    repeat_scen = []
    for h in hypothesis:
        for d in detection_ids:
            repeat_scen.append(h + [d])

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
        for ed in np.where(tree_compat[i] ==1)[0]:
            edges.append((i,ed))


    g.vs["label"] = labels

    g.add_edges(edges)
    hyps =  g.maximal_independent_vertex_sets()
    return hyps


def update_hyps_scores_multi(dets, hyps, scores, probz, bg_prob):
    new_hyps = []
    new_scores = []

    detection_ids = dets["id"].to_list()

    for hyp_i, hyp in enumerate(hyps):
        existing_score = scores[hyp_i]

        # for h in hyp:
        branches = make_branches_multi(hyp, detection_ids)
        br_scores = score_new_branches(branches, bg_prob, probz)

        hyp_inds = make_hypotheses_from_trees(branches)

        for hind in hyp_inds:
            new_hyps.append([branches[i] for i in hind])

            new_scores.append(existing_score + np.sum([br_scores[i] for i in hind]))

    return new_hyps, new_scores

if __name__ == "__main__":



    import pandas as pd
    import matplotlib.pyplot as plt


    detections = pd.DataFrame()
    detections["id"] =              [0, 1, 2, 3, 4, 5, 6]
    detections["time"] =            [0, 1, 1, 2, 4, 4, 6]
    detections["object_location"] = [0, 1, 2, 3, 4, 5, 6]

    fauna_step = 0.2
    background_prob = 0.2

    hyps, scores = build_and_score_hypotheses_multi(detections, fauna_step, background_prob)

    for i, h in enumerate(hyps):
        print(h, scores[i])

    fig, ax = plt.subplots()
    ax.scatter([len(h) for h in hyps], scores) #[scores[i]/len(h) for i, h in enumerate(hyps)] )
    plt.show()

    a=10

