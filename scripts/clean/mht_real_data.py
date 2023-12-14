import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis, euclidean
import igraph
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

class LobsterDetections:
    def __init__(self,
                 csv_path='../../data/detections_for_mht.csv'):
        if type(csv_path) == str:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = csv_path
        self.detections = self.make_detections_df()

    def make_detections_df(self):
        cols = ['time',
                'x',
                'y',
                'id']
        df = pd.DataFrame(columns=cols)
        df.time = self.df.outtime
        df.x = self.df.X
        df.y = self.df.Y
        df.id = self.df.detection_id

        return df


class LobsterMHT:
    def __init__(self,
                 detections,
                 cam_area=[1.3,1.6],
                 density=0.03,
                 var=0.1,
                 thresh=0.5,
                 dist_thresh=2,
                 time_thresh=600):
        self.detections = detections
        self.density = density
        self.cam_area = cam_area
        self.var = var
        self.thresh = thresh
        self.dist_thresh = dist_thresh
        self.time_thresh = time_thresh
        self.pmiss = 1/(9*var)#pmiss
        self.bg_prob = self.density / np.multiply(*self.cam_area)

    def make_first_track(self):
        df = self.detections.copy()
        t = df.loc[0].time
        det = df[df.time == t]

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
                time_del = self.time_between_dets(this, last)
                if time_del <= self.time_thresh:
                    if mah <= self.thresh:
                        dist = self.dist_between_dets(this, last)
                        if dist <= self.dist_thresh:
                            new_tracks.append(repeat_scen)
                            new_scores.append(score)

        return new_tracks, new_scores

    def dist_between_dets(self, this, last):
        # get the xy position of the current and previous detection
        thispos = self.detections.loc[this][["x", "y"]].to_numpy()
        lastpos = self.detections.loc[last][["x", "y"]].to_numpy()
        return euclidean(thispos, lastpos)

    def time_between_dets(self, this, last):
        thist = self.detections.loc[this].time
        lastt = self.detections.loc[last].time
        return thist - lastt

    def score_between_dets(self, this, last):
        # get the xy position of the current and previous detection
        thispos = self.detections.loc[this][["x", "y"]].to_numpy()
        lastpos = self.detections.loc[last][["x", "y"]].to_numpy()

        # get the time of the detections
        thist = self.detections.loc[this].time
        lastt = self.detections.loc[last].time

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

def get_last(hyp):
    h = hyp
    tlast = (~np.isnan(h)).cumsum().argmax()
    return hyp[tlast]

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


def delete_by_index(mylist, indices):
    for index in sorted(indices, reverse=True):
        del mylist[index]


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

def plot_hyp_scores(hyps,
                    hscores,
                    cluster,
                    n_detections):
    hlen = [len(h) for h in hyps]
    fig, ax = plt.subplots()
    ax.scatter(hlen, hscores)
    ax.set_xlabel("Predicted number of targets")
    ax.set_ylabel("Hypothesis score")
    ax.set_yticks([])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle('Cluster {}: {} detections'.format(cluster, n_detections))
    plt.show()


def get_final_tracks(best_tracks, df_all):

    out = []
    # best track is a list of lists, one for each cluster
    for bt in best_tracks:
        for tr in bt:
            X = []
            Y = []

            for det in tr:
                row = df_all.loc[det]
                X.append(row["X"])
                Y.append(row["Y"])
                cluster = row["CLUSTER_ID"]
            out.append(dict(X=X,
                            Y=Y,
                            cluster=cluster))
    return pd.DataFrame(out)

def get_final_tracks_shp(best_tracks, df_all):

    out = []
    # best track is a list of lists, one for each cluster
    i=0
    for bt in best_tracks:
        for tr in bt:
            for det in tr:
                row = df_all.loc[det]
                out.append(dict(longitude=row["longitude"],
                            latitude=row["latitude"],
                            trackid=i,
                            cluster=row["CLUSTER_ID"]))
            i+=1

    return pd.DataFrame(out)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df_all = pd.read_csv('../../data/detections_for_mht-clustered.csv')
    df_all["detection_id"] = range(len(df_all))
    plt.scatter(df_all.X, df_all.Y, c=df_all.CLUSTER_ID)
    plt.show()

    clusters = df_all.CLUSTER_ID.unique()

    best_tracks = []
    for cluster in clusters:
        df = df_all[df_all.CLUSTER_ID==cluster].reset_index()
        ld = LobsterDetections(df)
        detections = ld.detections.copy()

        lm = LobsterMHT(detections)

        tracks = lm.make_first_track()
        scores = [np.log(lm.bg_prob)] * len(tracks)

        tsteps = detections.time.unique()
        print("Making tracks")
        # for tstep in tqdm(tsteps[1:]):
        #     det = detections[detections.time==tstep]
        #     tracks, scores =  lm.update_tracks(det, tracks, scores)

        k = 5
        for tstep in tsteps[1:]:
            # tstep = tsteps[i]
            det = detections[detections.time == tstep]
            tracks, scores = lm.update_tracks(det, tracks, scores)
            if len(tracks[0]) > k:

                pi = prune(tracks, scores, k)
                if len(pi) > 0:
                    # print(i, len(tracks))
                    # print("Pruning tracks {}".format(pi))
                    delete_by_index(tracks, pi)
                    delete_by_index(scores, pi)
        print("Cluster {}".format(cluster))
        print("# tracks = ", len(tracks))
        print("building final hyps")
        hyps, hscores = build_hypotheses(tracks, scores)
        print('# hyps = ', len(hyps))

        plot_hyp_scores(hyps, hscores, cluster, len(detections))
        # alltracks.append(tracks)
        global_hyp = hyps[np.argmax(hscores)]
        global_hyp_tracks = [tracks[h] for h in global_hyp]

        # convert ids to actual detection ids (they start from zero for each cluster)
        out_tracks = []
        for tr in global_hyp_tracks:
            tr = [t for t in tr if not np.isnan(t)]
            out_tracks.append([df.loc[i].detection_id for i in tr if not np.isnan(i)])

        best_tracks.append(out_tracks)

    clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']
    trackdf = get_final_tracks(best_tracks, df_all)

    clusters.sort()
    fig, ax = plt.subplots()
    ax.scatter(df_all["X"], df_all["Y"], facecolors='none', edgecolors=[clrs[c] for c in df_all.CLUSTER_ID])
    for i, row in trackdf.iterrows():
        ax.plot(row["X"], row["Y"],c=clrs[row["cluster"]], linewidth=3)
    ax.set_xticks([])
    ax.set_yticks([])
    # fig.legend(clusters, title='Cluster')
    fig.show()

    # ld = LobsterDetections()
    # plt.scatter(ld.df.X, ld.df.Y)
    # plt.show()
    #
    # detections = ld.detections.copy()
    #
    # lm = LobsterMHT(detections)
    #
    # tracks = lm.make_first_track()
    # scores = [np.log(lm.bg_prob)]*len(tracks)
    #
    # tsteps = detections.time.unique()
    # print("Making tracks")
    # # for tstep in tqdm(tsteps[1:]):
    # #     det = detections[detections.time==tstep]
    # #     tracks, scores =  lm.update_tracks(det, tracks, scores)
    #
    # k = 5
    # for i in range(1,30):
    #     tstep = tsteps[i]
    #     det = detections[detections.time == tstep]
    #     tracks, scores = lm.update_tracks(det, tracks, scores)
    #     if len(tracks[0])>k:
    #         if i ==i :
    #             pi = prune(tracks, scores, k)
    #             if len(pi)>0:
    #
    #                 print(i, len(tracks))
    #                 print("Pruning tracks {}".format(pi))
    #                 delete_by_index(tracks, pi)
    #                 delete_by_index(scores, pi)
    #
    # print("# tracks = ", len(tracks))
    # print("building final hyps")
    # hyps, hscores = build_hypotheses(tracks, scores)
    # print('# hyps = ', len(hyps))
