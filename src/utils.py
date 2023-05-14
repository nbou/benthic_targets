import numpy as np
from scipy import stats

def two_tail_prob(z):
    return 2*(1-stats.norm.cdf(z))

def get_last_detection_id(br):
    return np.array(br)[np.logical_not(np.isnan(br))][-1]



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
