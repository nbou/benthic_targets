'''
1. Generate vehicle track (to figure out number of timesteps to simulate
2. Generate simulated movement of targets
3. Generate detections (i.e. when targets are in camera)
4. Use detections to estimate distribution with alg
5. Analyse accuracy of distribution estimate
'''

from src.simulate import make_positions, mow_the_lawn_2D, get_in_camera_dets
from src.build_gate_rank_hypotheses import build_and_score_hypotheses_multi

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class ConfigSim:
    def __init__(self,
                 survey_width,
                 n_fauna,
                 fauna_speed,
                 camera_footprint,
                 camera_step,
                 camera_ystep,
                 bg_prob):
        self.survey_width = survey_width
        self.n_fauna = n_fauna
        self.fauna_speed = fauna_speed
        self.camera_footprint = camera_footprint
        self.camera_step = camera_step
        self.camera_ystep = camera_ystep
        self.background_probability = bg_prob


class Simulate:
    def __init__(self, conf: ConfigSim):
        self.Config = conf
        self.camx, self.camy = mow_the_lawn_2D(0, 0, conf.survey_width, conf.survey_width, conf.camera_step,
                                               conf.camera_ystep)
        self.animals = make_positions(conf.n_fauna, conf.fauna_speed, conf.survey_width, len(self.camx))
        self.detections = get_in_camera_dets(self.animals, self.camx, self.camy, conf.camera_footprint)

    def alg_from_dets(self):
        return build_and_score_hypotheses_multi(
            self.detections,
            self.Config.fauna_speed,
            self.Config.background_probability,
            0.05
        )

    def plot_sim(self):
        n_tsteps = len(self.camx)

        markersize = 20
        clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots(1, 2)
        # plot just animal positions
        stp = 10
        for i in self.animals["individual"]:
            print(i)
            ax[0].scatter(self.animals.loc[i][["x_{}".format(i) for i in range(1, n_tsteps + 1, stp)]].to_numpy(),
                          self.animals.loc[i][["y_{}".format(i) for i in range(1, n_tsteps + 1, stp)]].to_numpy(),
                          marker='.',
                          s=markersize
                          , c=clrs[i])
            ax[0].scatter(self.animals.loc[i]["x_0"], self.animals.loc[i]["y_0"], marker='+', color='black', s=100)

        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # add camera path
        ax[0].plot(self.camx, self.camy)
        ax[0].axis('square')
        ax[0].axis([-1, self.Config.survey_width + 1, -1, self.Config.survey_width + 1])

        for t in self.detections["time"].unique():
            tdf = self.detections[self.detections["time"] == t]

            ax[1].add_patch(Rectangle(
                (tdf["camera_x"].to_numpy()[0] - 0.5 * self.Config.camera_footprint,
                 tdf["camera_y"].to_numpy()[0] - 0.5 * self.Config.camera_footprint),
                self.Config.camera_footprint, self.Config.camera_footprint,
                facecolor=(0.1, 0, .8, 0.05),
                edgecolor=(0.1, 0, .8, 0.5)))

        for ind in range(self.Config.n_fauna):  # , ind in enumerate(detections["individual"].unique()):
            # print("2", ind)

            idf = self.detections[self.detections["individual"] == ind]
            if len(idf) > 0:
                ax[1].scatter(idf["x"], idf[["y"]], marker='.', s=markersize, c=clrs[ind])

        ax[1].set_xticks([])
        ax[1].set_yticks([])

        ax[1].axis('square')
        ax[1].axis([-1, self.Config.survey_width + 1, -1, self.Config.survey_width + 1])

        fig.tight_layout()
        return fig


if __name__ == "__main__":
    area = 10
    n_fauna = 10
    speed = 0.1
    cam_footprint = 1
    cam_step = 0.3
    cam_ystep = 2

    bg_prob = n_fauna / (area * area)

    config = ConfigSim(area,
                       n_fauna,
                       speed,
                       cam_footprint,
                       cam_step,
                       cam_ystep,
                       bg_prob)

    sim = Simulate(config)

    fig = sim.plot_sim()
    fig.show()

    hyps, scores = sim.alg_from_dets()
