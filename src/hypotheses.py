import numpy as np
from scipy import stats

def two_tail_prob(z):
    return 2*(1-stats.norm.cdf(z))

def get_last_detection_id(br):
    return np.array(br)[np.logical_not(np.isnan(br))][-1]

# function to update hypotheses and scores where there are more than one detections at the new timestep
def update_hyps_scores_multi(det, hyps, scores, probz, bg_prob):
    new_hyps = []
    new_scores = []


    # go through each existing hypothesis
    for hyp_i, hyp in enumerate(hyps):
        existing_score = scores[hyp_i]

        # scenario where all the detections are new individuals
        new_scen = [h + [np.nan] for h in hyp] + [list(np.empty(len(hyp[0])) * np.nan) + [d] for d in det["id"].to_list()]
        new_hyps.append(new_scen)
        # score is the old score plus number of detections X the bg probability
        scr = existing_score + bg_prob * len(det)
        new_scores.append(scr)


        # go through each new detection and append the case where that detection is a repeat of the current hypothesis
        for detection in det:



def update_hyps_scores(det, hyps, scores, probz, bg_prob):
    new_hyps = []
    new_scores = []

    for i, row in det.iterrows():
        detection_id = row["id"]
        detection_time = row["time"]
        detection_pos = row["object_location"]

        #     detection_id = det.iloc[0].id
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


def generate_hypothesis_matrix(detections, faunastep):

    n_dets = len(detections)
    dists = np.zeros((len(detections), len(detections)))
    Dtime = np.zeros((len(detections), len(detections)))
    for i in range(n_dets):
        for j in range(n_dets):
            dists[i, j] = np.abs(detections.loc[i]["object_location"] - detections.loc[j]["object_location"])
            Dtime[i, j] = np.abs(detections.loc[i]["time"] - detections.loc[j]["time"])

    z_stat = np.divide(dists, faunastep * Dtime)
    return two_tail_prob(z_stat)


def build_and_score_hypotheses(detections, faunastep, bg_prob):
    probz = generate_hypothesis_matrix(detections, faunastep)

    # make hypotheses and scores for first timestep
    scores = []
    hyps = []
    for i, row in detections[detections["time"] == 0].iterrows():
        hyps.append([[row["id"]]])
        scores.append(scores.append(bg_prob))

    # update hypotheses and scores for the subsequent timesteps
    for t in detections.time.unique()[1:]:
        hyps, scores = update_hyps_scores(detections[detections["time"]==t],
                                          hyps,
                                          scores, probz,
                                          bg_prob)

    return hyps, scores

if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt

    detections = pd.DataFrame()
    detections["id"] = [0, 1, 2, 3, 4]
    detections["time"] = [0, 1, 1, 2, 3]
    detections["object_location"] = [0, 1, 2, 3, 4]

    fauna_step = 0.2
    background_prob = 0.2

    hyps, scores = build_and_score_hypotheses(detections, fauna_step, background_prob)

    fig, ax = plt.subplots()
    ax.scatter([len(h) for h in hyps], scores)
    fig.show()

    a=10

