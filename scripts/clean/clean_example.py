import numpy as np
from src.simulate import initial_positions
from src.utils import get_last_detection_id

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pandas as pd
import itertools
import igraph
from scipy.spatial.distance import mahalanobis

def simulate2(starting, norm):
    return [s + np.random.normal(0, norm, 2) for s in starting]

def simulate1(starting, norm):
    return [s + np.random.normal(0, norm) for s in starting]

def simulate(starting, norm, n_steps, d2=False):
    s = [starting]
    for i in range(n_steps):
        if d2:
            s.append(simulate2(s[i], norm))
        else:
            s.append(simulate1(s[i], norm))

    return np.array(s)

def in_cam(cam_x, cam_y, t, cam_area, positions):
    cx = cam_x[t]
    cy = cam_y[t]

    pos = positions[t]

    dets = []

    for p in pos:
        inx = np.logical_and(cx <= p[0], p[0] <= cx + cam_area[0])
        iny = np.logical_and(cy <= p[1], p[1] <= cy + cam_area[1])
        dets.append([np.logical_and(inx, iny)])
    return dets


class conceptExample:
    def __init__(self,
                 survey_x=5,
                 survey_y=1,
                 n_inds=3,
                 cam_area=[1,1],
                 cam_step=0.6,
                 var=0.1,
                 thresh=0.3,
                 pmiss=0.3
                 ):
        self.survey_x = survey_x
        self.survey_y = survey_y
        self.n_inds = n_inds
        self.cam_area = cam_area
        self.cam_step = cam_step
        self.var = var
        self.thresh = thresh
        self.pmiss = pmiss

        self.density = self.n_inds/(self.survey_y*self.survey_x)
        self.bg_prob = self.density/np.multiply(*self.cam_area)

        self.starting_x, self.starting_y = self.make_starting()
        self.cam_x, self.cam_y = self.make_campath()

        self.n_steps = len(self.cam_x)
        self.positions = self.make_paths()

        self.detections = self.make_detections()

        self.df = self.make_df()
        self.tracks = self.make_first_track()

    def make_starting(self):
        area = [0, self.survey_x, 0, self.survey_y]
        starting_x, starting_y = initial_positions(self.n_inds, area)
        # starting_x = [.8, 2., 4.]
        return starting_x, starting_y

    def plot_starting(self):
        fig, ax = plt.subplots()

        for x, y in zip(self.starting_x, self.starting_y):
            ax.scatter(x, y)

        ax.axis('equal')
        ax.set_xlim(0 - .5, self.survey_x + .5)
        ax.set_ylim(0 - .5, self.survey_y + .5)

        ax.axis('off')

        ax.add_patch(Rectangle((0, 0), self.survey_x, self.survey_y, fill=False))

        plt.show()

    def make_campath(self):
        c_x = 0

        cam_x = [c_x]
        while c_x+self.cam_area[0] < self.survey_x:
            c_x += self.cam_step
            cam_x.append(c_x)

        cam_y = [0]*len(cam_x)


        return cam_x, cam_y

    def plot_cams(self):
        fig, ax = plt.subplots()
        for x, y in zip(self.cam_x[:3], self.cam_y[:3]):
            ax.add_patch(Rectangle((x, y), self.cam_area[0], self.cam_area[1], edgecolor='black', linewidth=2, alpha=0.7, zorder=0))

        for x, y in zip(self.starting_x, self.starting_y):
            ax.scatter(x, y)

        ax.axis('square')
        ax.set_xlim(0 - .5, self.survey_x + .5)
        ax.set_ylim(0 - .5, self.survey_y + .5)

        ax.axis('off')

        ax.add_patch(Rectangle((0, 0), self.survey_x, self.survey_y, fill=False))

        plt.show()

    def make_paths(self):
        starting = np.array([[x, y] for x, y in zip(self.starting_x, self.starting_y)])

        return simulate(starting, self.var, self.n_steps, d2=True)
    def plot_paths(self):


        fig, ax = plt.subplots()

        for i in range(self.n_inds):
            ax.plot(self.positions[:, i, 0], self.positions[:, i, 1], '-o', markersize=3)

        ax.plot(self.starting_x, self.starting_y, 'kx')
        ax.axis('square')
        ax.set_xlim(0 - .5, self.survey_x + .5)
        ax.set_ylim(0 - .5, self.survey_y + .5)

        ax.axis('off')

        ax.add_patch(Rectangle((0, 0), self.survey_x, self.survey_y, fill=False))

        plt.show()

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


    def plot_detections(self):
        clrs = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
        fig, ax = plt.subplots()

        for d in self.detections:
            t = d[0]
            ind = d[1]

            pos = self.positions[t][ind]
            ax.scatter(pos[0], pos[1], color=clrs[ind])
            ax.add_patch(
                Rectangle((self.cam_x[t], self.cam_y[t]), self.cam_area[0], self.cam_area[1], edgecolor='black', linewidth=2, alpha=0.4,
                          zorder=0))

        ax.axis('square')
        ax.set_xlim(0 - .5, self.survey_x + .5)
        ax.set_ylim(0 - .5, self.survey_y + .5)

        ax.axis('off')
        plt.show()

    def make_first_track(self):
        t = self.df.loc[0].time
        det = self.df[self.df.time == t]

        tracks = [[i] for i, row in det.iterrows()]

        # tracks.append([np.nan])
        return tracks

    def update_tracks(self, dets, tracks, scores):
        new_tracks = []
        new_scores = []
        for i, tr in enumerate(tracks):
            # scenario where it's a missed detection add nan to end of each track
            missed_scen = tr + [np.nan]
            new_tracks.append(missed_scen)

            # score here is pmiss
            new_scores.append(scores[i] + np.log(self.pmiss))

        for det in dets.index:
            # Scenario where it's a new individual
            nans = list(np.ones(len(tracks[0])) * np.nan)
            new_ind_scen = nans + [det]
            new_tracks.append(new_ind_scen)

            # score here is the background probability
            new_scores.append(scores[i] + np.log(self.bg_prob))

            for i, tr in enumerate(tracks):
                # association score from previous detection
                # get previous detection, calculate score
                repeat_scen = tr + [det]
                this = det
                last = get_last(tr)
                mah, score = self.score_between_dets(this, last)

                #gating
                if mah <= self.thresh:
                    new_tracks.append(repeat_scen)
                    new_scores.append(score)

        return new_tracks, new_scores

    def score_between_dets(self, this, last):
        # get the xy position of the current and previous detection
        thispos = self.df.loc[this][["x", "y"]].to_numpy()
        lastpos = self.df.loc[last][["x", "y"]].to_numpy()

        # get the time of the detections
        thist = self.df.loc[this].time
        lastt = self.df.loc[last].time

        # calculate the elapsed time between the detections
        dt = thist - lastt

        # build the covariance matrix and calculate the malahanobis distance between detections
        cov = np.array([[dt * self.var, 0], [0, dt * self.var]])
        d2 = mahalanobis(lastpos, thispos, cov)
        score = score_del(np.multiply(*self.cam_area), cov, d2)

        return d2, score


def score_del(area, cov, mah):
    delta = np.log(np.divide(area, 2*np.pi)) - 0.5*np.log(np.linalg.det(cov)) - 0.5*mah
    return delta
# def score_missed_scen(pmiss):
#     return pmiss
#
# def score_new_ind_scen(pnew):
#     return pnew
#
# def score_repeat_scen(last_detection, this_detection,)
def get_last(hyp):
    h = hyp
    tlast = (~np.isnan(h)).cumsum().argmax()
    return hyp[tlast]

# def update_tracks(dets, tracks):
#     new_tracks = []
#
#     for i, tr in enumerate(tracks):
#         # scenario where it's a missed detection add nan to end of each track
#         missed_scen = tr + [np.nan]
#         new_tracks.append(missed_scen)
#
#         # score here is pmiss
#
#     for det in dets.index:
#         # Scenario where it's a new individual
#         nans =  list(np.ones(len(tracks[0]))*np.nan)
#         new_ind_scen = nans + [det]
#         new_tracks.append(new_ind_scen)
#
#         # score here is the background probability
#
#         for i, tr in enumerate(tracks):
#             repeat_scen = tr + [det]
#             new_tracks.append(repeat_scen)
#
#             # association score from previous detection
#             # get previous detection, calculate score
#     return new_tracks

def score_hypotheses(hyps, scores):
    hscores = []
    for hyp in hyps:
        hscores.append(np.sum([scores[h] for h in hyp]))
    return hscores

def build_hypotheses(tracks, scores):
    gtrack = np.array(tracks)
    g = igraph.Graph()
    nverts = len(gtrack)
    g.add_vertices(nverts)

    edges = []

    for i, tr in enumerate(gtrack):
        diff = tr - gtrack
        for d in np.where(diff == 0)[0]:
            if d != i:
                edges.append((i, d))

    g.add_edges(edges)
    hyps = g.maximal_independent_vertex_sets()
    hscores = score_hypotheses(hyps, scores)

    return hyps, hscores


def first_track(dets):
    first_timestep = dets.loc[0].time
    det = dets[dets.time == first_timestep]

    tracks = [[i] for i, row in det.iterrows()]
    return tracks

if __name__ == "__main__":
    ce = conceptExample()

    ce.plot_starting()
    ce.plot_cams()
    ce.plot_paths()
    ce.plot_detections()


    detections = ce.df
    #
    # df = pd.DataFrame(columns = ['time', 'individual', 'x', 'y'])
    # df.time = [0, 1, 1, 3]
    # df.individual = [0, 0, 1, 2]
    # df.x = [0., 0.001, 2, 3]
    # df.y = [0.5, 0.5, 0.5, 0.5]

    tracks = ce.make_first_track()
    scores = [np.log(ce.bg_prob)]*len(tracks)

    tsteps = detections.time.unique()
    for tstep in tsteps[1:]:
        det = detections[detections.time==tstep]
        tracks, scores =  ce.update_tracks(det, tracks, scores)
        print('Timestep {}:'.format(tstep))



    hyps, hscores = build_hypotheses(tracks, scores)

    print(np.argsort(hscores))
    d = np.argsort(hscores)[-1]
    print(detections)
    print([tracks[h] for h in hyps[d]])
    a=10