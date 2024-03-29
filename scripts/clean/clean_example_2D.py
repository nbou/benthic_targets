
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from scipy.spatial.distance import mahalanobis

from clean_example import simulate, score_del, initial_positions, get_last, build_hypotheses, plot_hyp_scores


class conceptExample2d:
    def __init__(self,
                 survey_x=5,
                 survey_y=5,
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
        self.tracks = self.make_first_track()

    def make_starting(self):
        area = [0+1, self.survey_x-1, 0+1, self.survey_y-1]
        starting_x, starting_y = initial_positions(self.n_inds, area)

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
        fig.tight_layout()
        plt.show()

    def make_campath(self):
        cam_x, cam_y = mow_the_lawn_2D(0,0, self.survey_x, self.survey_y, self.cam_step)

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
        clrs = plt.rcParams['axes.prop_cycle'].by_key()['color'][:self.n_inds]
        fig, ax = plt.subplots()

        ax.plot(self.cam_x, self.cam_y)

        halfcam = np.divide(self.cam_area,2)
        for d in self.detections:
            t = d[0]
            ind = d[1]

            pos = self.positions[t][ind]
            ax.scatter(pos[0], pos[1], color=clrs[ind])
            ax.add_patch(
                Rectangle((self.cam_x[t] - halfcam[0], self.cam_y[t]-halfcam[1]), self.cam_area[0], self.cam_area[1], edgecolor='black', linewidth=2, alpha=0.4,
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

def find_correct(hyps, hscores, tracks, detections):

    # drop nans from tracks
    trx = []
    for tr in tracks:
        trx.append([t for t in tr if np.isnan(t) == False])


    # get correct hyp
    correct = []
    for i in detections.individual.unique():
        correct.append(detections[detections.individual==i].index.to_list())

    print(correct)
    for i, hyp in enumerate(hyps):
        hypt = [trx[h] for h in hyp]
        if hypt == correct:
            hs = hscores[i]
            print(sum(s>hs for s in hscores)/len(hscores))






if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ce = conceptExample2d()

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

    plot_hyp_scores(hyps, hscores)

    find_correct(hyps, hscores, tracks, detections)
    # camx, camy = mow_the_lawn_2D()
    #
    # fig, ax = plt.subplots()
    # ax.plot(camx, camy)
    #
    # footprint=[1,1]
    # for i, x in enumerate(camx):
    #     if i%1 ==0:
    #         y = camy[i]
    #         ax.add_patch(Rectangle((x - 0.5 * footprint[0], y - 0.5 * footprint[1]), footprint[0], footprint[1], linewidth=3, edgecolor='blue',alpha=.5, facecolor='none'))
    #
    # plt.show()