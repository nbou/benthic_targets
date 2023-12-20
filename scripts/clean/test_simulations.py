# script to test performance of on simulated data
import numpy as np
import pandas as pd

from matplotlib.patches import Rectangle
from scipy.spatial.distance import mahalanobis

from clean_example import simulate, score_del, initial_positions, get_last, build_hypotheses, plot_hyp_scores

import multiprocessing
from functools import partial
from contextlib import contextmanager

class conceptExample2d:
    def __init__(self,
                 survey_x=5,
                 survey_y=3,
                 n_inds=3,
                 cam_area=[1,1],
                 cam_step=0.6,
                 var=0.1,
                 thresh=2,
                 # pmiss=1
                 ):
        self.survey_x = survey_x
        self.survey_y = survey_y
        self.n_inds = n_inds
        self.cam_area = cam_area
        self.cam_step = cam_step
        self.var = var
        self.thresh = thresh
        self.pmiss = 1/(20*var)#pmiss

        self.density = self.n_inds/(self.survey_y*self.survey_x)
        self.bg_prob = self.density/np.multiply(*self.cam_area)

        self.starting_x, self.starting_y = self.make_starting()
        self.cam_x, self.cam_y = self.make_campath()

        self.n_steps = len(self.cam_x)
        self.positions = self.make_paths()

        self.detections = self.make_detections()

        self.df = self.make_df()


    def make_starting(self):
        area = [0., self.survey_x, 0., self.survey_y]
        starting_x, starting_y = initial_positions(self.n_inds, area)
        starting_x = [1., 2.5, 3.4]
        return starting_x, starting_y


    def make_campath(self):
        cam_x, cam_y = mow_the_lawn_2D(0,0, self.survey_x, self.survey_y, self.cam_step)

        return cam_x, cam_y

    def make_paths(self):
        starting = np.array([[x, y] for x, y in zip(self.starting_x, self.starting_y)])

        return simulate(starting, self.var, self.n_steps, d2=True)


    def make_detections(self):
        detections = []

        for t in range(self.n_steps):
            d = in_cam(self.cam_x, self.cam_y, t, self.cam_area, self.positions)
            # each detection is [timestep, individual]
            npw = np.where(d)
            if len(npw[0]) > 0:
                for ind in npw[0]:
                    detections.append([t, ind])
        return detections

    def make_df(self):
        det_arr = np.array(self.detections)
        df = pd.DataFrame(columns=['time', 'individual', 'x', 'y'])
        df["time"] = det_arr[:, 0]
        df["individual"] = det_arr[:, 1]

        df["x"] = df.apply(lambda row: self.positions[row["time"], row["individual"], 0], axis=1)
        df["y"] = df.apply(lambda row: self.positions[row["time"], row["individual"], 1], axis=1)
        return df

def mow_the_lawn_2D(startx=0, starty=0, endx=5, endy=3, camstep=0.6, ysteps=1):
    # first leg, x increasing to max
    x = startx
    y = starty

    pathX = [startx]
    pathY = [starty]

    while y < endy-camstep:
        # Take steps in +ve x dir
        while x < endx:
            # take step in +ve x direction
            x += camstep
            # add new position to
            pathX.append(x)
            pathY.append(y)

        # Take ysteps steps in y dir
        i = 0
        while i < ysteps:
            i += 1
            y += camstep
            pathX.append(x)
            pathY.append(y)

        # start going back in -ve x direction
        while x > startx:
            x -= camstep
            # add new position to
            pathX.append(x)
            pathY.append(y)

            # Take ysteps steps in y dir
        i = 0
        while i < ysteps:
            i += 1
            y += camstep
            pathX.append(x)
            pathY.append(y)

    # do last leg in X directions to end
    while x < endx:
        # take step in +ve x direction
        x += camstep
        # add new position to
        pathX.append(x)
        pathY.append(y)
    return pathX, pathY

def in_cam(cam_x, cam_y, t, cam_area, positions):
    cx = cam_x[t]
    cy = cam_y[t]

    pos = positions[t]

    dets = []

    halfcam = np.divide(cam_area, 2)
    for p in pos:
        inx = np.logical_and(cx - halfcam[0] <= p[0], p[0] <= cx + halfcam[1])
        iny = np.logical_and(cy - halfcam[0] <= p[1], p[1] <= cy + halfcam[1])
        dets.append([np.logical_and(inx, iny)])
    return dets

def score_sim(hyps, hscores, tracks, detections, n_individuals=3):

    # drop nans from tracks
    trx = []
    for tr in tracks:
        trx.append([t for t in tr if np.isnan(t) == False])


    # get correct hyp
    correct = []
    for i in detections.individual.unique():
        correct.append(detections[detections.individual==i].index.to_list())

    # find percentage of scores higher than the correct one
    hs = min(hscores)
    for i, hyp in enumerate(hyps):
        hypt = [trx[h] for h in hyp]
        if hypt == correct:
            hs = hscores[i]
            break
    score_percentile = sum(s > hs for s in hscores) / len(hscores)

    # get top 10% scoring hypotheses
    n10 = int(np.ceil(0.1*len(hscores)))
    i10 = np.argsort(hscores)[::-1][:n10]

    top_hyps = [hyps[i] for i in i10]
    prop_correct_abundance = sum(len(t)==n_individuals for t in top_hyps)/n10

    return score_percentile, prop_correct_abundance




def make_first_track(df):
    t = df.loc[0].time
    det = df[df.time == t]

    tracks = [[i] for i, row in det.iterrows()]

    # tracks.append([np.nan])
    return tracks



def score_between_dets(df, this, last, var, cam_area):
    # get the xy position of the current and previous detection
    thispos = df.loc[this][["x", "y"]].to_numpy()
    lastpos = df.loc[last][["x", "y"]].to_numpy()

    # get the time of the detections
    thist = df.loc[this].time
    lastt = df.loc[last].time

    # calculate the elapsed time between the detections
    dt = thist - lastt

    # build the covariance matrix and calculate the malahanobis distance between detections
    cov = np.array([[dt * var, 0], [0, dt * var]])
    d2 = mahalanobis(lastpos, thispos, cov)
    score = score_del(np.multiply(*cam_area), cov, d2)

    return d2, score

class SimConfig:
    def __init__(self,
                 var=0.1,
                 cam_area= [1, 1],
                 n_inds = 3,
                 density=3./15,
                 thresh=2
                 ):
        self.var = var
        self.thresh = thresh
        self.pmiss = 1/(9*var)#pmiss
        self.cam_area = cam_area
        self.n_inds=n_inds
        self.density = density #self.n_inds/(self.survey_y*self.survey_x)
        self.bg_prob = self.density/np.multiply(*self.cam_area)


def update_tracks(dets, all_dets, tracks, scores, conf:SimConfig):
    new_tracks = []
    new_scores = []
    for i, tr in enumerate(tracks):
        # scenario where it's a missed detection add nan to end of each track
        missed_scen = tr + [np.nan]
        new_tracks.append(missed_scen)

        # score here is pmiss
        new_scores.append(scores[i] + np.log(conf.pmiss))

    for det in dets.index:
        # Scenario where it's a new individual
        nans = list(np.ones(len(tracks[0])) * np.nan)
        new_ind_scen = nans + [det]
        new_tracks.append(new_ind_scen)

        # score here is the background probability
        new_scores.append(scores[i] + np.log(conf.bg_prob))

        for i, tr in enumerate(tracks):
            # association score from previous detection
            # get previous detection, calculate score
            repeat_scen = tr + [det]
            this = det
            last = get_last(tr)
            mah, score = score_between_dets(all_dets, this, last, conf.var, conf.cam_area)

            #gating
            if mah <= conf.thresh:
                new_tracks.append(repeat_scen)
                new_scores.append(score)

    return new_tracks, new_scores


def check_detections(df, target_num=3, maxlen=14):
    out = True
    if len(df)>maxlen:
        out=False

    n_inds = df.individual.nunique()
    if n_inds<target_num:
        out = False

    return out

def prune(tracks, scores, k):
    hyps, hscores = build_hypotheses(tracks, scores)
    hyp = [tracks[h] for h in hyps[np.argmax(hscores)]]

    prune_inds = []

    for tr in hyp:
        track_stem = tr[:-(k - 1)]

        for i, track in enumerate(tracks):
            if (track != tr) & (track[:-(k - 1)] == track_stem):
                prune_inds.append(i)
    return prune_inds

def delete_by_index(mylist, indices):
    for index in sorted(indices, reverse=True):
        del mylist[index]

def make_tracks_hyps(detections, conf:SimConfig):
    tracks = make_first_track(detections)
    scores = [np.log(conf.bg_prob)]*len(tracks)

    k=3
    tsteps = detections.time.unique()
    for tstep in tsteps[1:]:
        det = detections[detections.time==tstep]
        tracks, scores =  update_tracks(det, detections, tracks, scores, conf)
        # print('Timestep {}:'.format(tstep))
        if len(tracks[0]) > k:

            pi = prune(tracks, scores, k)
            if len(pi) > 0:
                # print(i, len(tracks))
                # print("Pruning tracks {}".format(pi))
                delete_by_index(tracks, pi)
                delete_by_index(scores, pi)
    hyps, hscores = build_hypotheses(tracks, scores)
    return tracks, hyps, hscores



@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def do_scoring(detections, conf:SimConfig):
    tracks, hyps, hscores = make_tracks_hyps(detections, conf)
    sp, ap = score_sim(hyps, hscores, tracks, detections, n_individuals=3)
    return [sp, ap]



if __name__ == "__main__":
    from tqdm import tqdm


    cfg = SimConfig()

    n_sims=10
    detlist = []
    while len(detlist) < n_sims:

        ce = conceptExample2d()
        detections = ce.df
        if check_detections(detections):
            detlist.append(ce.df)



    with poolcontext(processes=10) as pool:
        results = pool.map(partial(do_scoring,conf=cfg), detlist)

    print(np.average(np.array(results)[:, 0]))
    print(np.average(np.array(results)[:,1]))

    # score_percs = []
    # abundance_props = []
    #
    #
    #
    # for dtc in tqdm(detlist):
    #     print(len(dtc))
    #     tracks, hyps, hscores = make_tracks_hyps(dtc, cfg)
    #     sp, ap = score_sim(hyps, hscores, tracks, dtc, n_individuals=3)
    #     score_percs.append(sp)
    #     abundance_props.append(ap)
    #
    # print(np.average(score_percs), np.average(abundance_props))