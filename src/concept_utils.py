import numpy as np
from src.simulate import initial_positions

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pandas as pd

from scipy.spatial.distance import mahalanobis

class conceptExample:
    def __init__(self,
                 survey_x=5,
                 survey_y=1,
                 n_inds=3,
                 cam_area=[1,1],
                 cam_step=0.6,
                 var=0.1,
                 thresh=1,
                 ):
        self.survey_x = survey_x
        self.survey_y = survey_y
        self.n_inds = n_inds
        self.cam_area = cam_area
        self.cam_step = cam_step
        self.var = var
        self.thresh = thresh
        self.density = self.n_inds/(self.survey_y*self.survey_x)

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
        starting_x = [.8, 2., 4.]
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

        ax.add_patch(Rectangle((0, 0), self.survey_x, self.survey_y, fill=False))

        plt.show()

    def make_first_track(self):
        t = 0
        det = self.df[self.df.time == t]

        tracks = [[i] for i, row in det.iterrows()]
        tracks.append([np.nan])
        return tracks

    def update_tracks(self, det, tracks):
        new_tracks = []
        for i, hyp in enumerate(tracks):
            missed_scen = hyp + [np.nan]

            new_tracks.append(missed_scen)

            for detection_id in det.index:
                repeat_scen = hyp + [detection_id]

                # do the gating
                gate = gating(repeat_scen, self.df, self.var, self.thresh)
                if gate == "keep":
                    new_tracks.append(repeat_scen)
                else:
                    print("Gated detection {} from {}".format(det.index[0], get_last(repeat_scen)))
        return new_tracks

    def update_score_tracks(self, det, tracks, scores):
        pmiss = self.density * np.multiply(*self.cam_area)
        new_tracks = []
        new_scores = []
        for i, hyp in enumerate(tracks):
            old_score = scores[i]
            missed_scen = hyp + [np.nan]

            new_tracks.append(missed_scen)
            new_scores.append(pmiss + old_score)

            for detection_id in det.index:
                repeat_scen = hyp + [detection_id]

                # do the gating
                gate, score = gate_score(repeat_scen, self.df, self.var, self.thresh, np.multiply(*self.cam_area) , self.density)

                if gate == "keep":
                    new_tracks.append(repeat_scen)
                    new_scores.append(score+ old_score)
                else:
                    print("Gated detection {} from {}".format(det.index[0], get_last(repeat_scen)))
        return new_tracks, new_scores




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


def get_last(hyp):
    h = hyp[:-1]
    tlast = (~np.isnan(h)).cumsum().argmax()
    return hyp[tlast]


# decide whether a track should be discarded or not (based on mahalanobis dist)
# should only run this if neither the current and previous detection aren't nans
def gating(track, df, var, thresh):
    # get the id of the current and previous detection in the track
    this = track[-1]
    #     print(this)
    last = get_last(track)

    # if either this or the previous detections are nans then keep the track
    if np.logical_or(np.isnan(this), np.isnan(last)):
        return "keep"

    else:
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
        #         print("mahal = ", d2)
        # if
        if d2 > thresh:
            return "discard"
        else:
            return "keep"


def gate_score(track, df, var, thresh, cam_area, density):
    # get the id of the current and previous detection in the track
    this = track[-1]
    #     print(this)
    last = get_last(track)

    # TODO: score for if this detection is a nan (i.e. missed detection) 0?
    # TODO: score for if previous is a nan (i.e. new individual) try inds/area * cam area?
    # if either this or the previous detections are nans then keep the track
    if np.isnan(last):
        gate="keep"
        delt=0
    elif np.isnan(this):
        gate="keep"
        delt = density * cam_area

    else:
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
        #         print("mahal = ", d2)
        # if
        if d2 > thresh:
            gate="discard"
            delt=None
        else:
            # TODO: calculate score delta and return this with the "keep"
            gate="keep"
            delt = score_del(cam_area, cov, d2)

    return gate, delt

def score_del(area, cov, mah):
    delta = np.log(np.divide(area, 2*np.pi)) - 0.5*np.log(np.linalg.det(cov)) - 0.5*mah
    return delta
