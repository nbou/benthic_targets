import pandas as pd
import numpy as np

from src.simulate import make_positions, mow_the_lawn_2D

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def check_in_cam(tstep, cx, cy, fprint, dfin):
    hfp = fprint / 2
    cxt = cx[tstep]
    cyt = cy[tstep]
    return dfin[(dfin["x_{}".format(tstep)].between(cxt - hfp, cxt + hfp)) & (
        dfin["y_{}".format(tstep)].between(cyt - hfp, cyt + hfp))][
        ["individual", "x_{}".format(tstep), "y_{}".format(tstep)]]


def get_in_camera_dets(df, camx, camy, footprint):
    out = []

    for i in range(len(camx)):
        cdf = check_in_cam(i, camx, camy, footprint, df)
        if len(cdf)!=0:
            for ind, row in cdf.iterrows():
                out.append(dict(time=i,
                                individual=row["individual"],
                                x=row["x_{}".format(i)],
                                y=row["y_{}".format(i)],
                                camera_x=camx[i],
                                camera_y=camy[i]
                                ))

    return pd.DataFrame(out)

def make_plots( n_inds = 10,
                   sze = 5,
                   speed = 0.1,
                   tsteps = 1000,
                   cam_fprint = 1,
                   cam_step = 0.3,
                   y_steps = 2):

    df = make_positions(n_inds, speed, sze, tsteps)
    camx, camy = mow_the_lawn_2D(0, 0, sze, sze, cam_step, y_steps)

    # generate detections (i.e. where individual is in a camera frame)

    detections = get_in_camera_dets(df, camx, camy, cam_fprint)

    # make plots

    # get number of timesteps in camera track
    n_tsteps = len(camx)


    markersize = 20
    clrs =  plt.rcParams['axes.prop_cycle'].by_key()['color']


    fig0, ax = plt.subplots()
    # plot just animal positions
    stp = 10
    for i in df["individual"]:
        print(i)
        ax.scatter(df.loc[i][["x_{}".format(i) for i in range(1, n_tsteps+1, stp)]].to_numpy(),
                  df.loc[i][["y_{}".format(i) for i in range(1, n_tsteps+1, stp)]].to_numpy(), marker='.', s=markersize
                      , c=clrs[i])
        ax.scatter(df.loc[i]["x_0"], df.loc[i]["y_0"], marker='+', color='black',s=100)

    ax.set_xticks([])
    ax.set_yticks([])

    # add camera path
    ax.plot(camx, camy)
    ax.axis('square')
    ax.axis([-1, sze + 1, -1,sze + 1])
    fig0.tight_layout()


    fig1, ax = plt.subplots()
    for t in detections["time"].unique():
        tdf = detections[detections["time"] == t]

        ax.add_patch(Rectangle(
            (tdf["camera_x"].to_numpy()[0] - 0.5 * cam_fprint, tdf["camera_y"].to_numpy()[0] - 0.5 * cam_fprint),
            cam_fprint, cam_fprint,
            facecolor=(0.1, 0, .8, 0.05),
            edgecolor=(0.1, 0, .8, 0.5)))

    for ind in range(n_inds):#, ind in enumerate(detections["individual"].unique()):
        # print("2", ind)

        idf = detections[detections["individual"] == ind]
        if len(idf) > 0:
            ax.scatter(idf["x"], idf[["y"]], marker='.', s=markersize, c=clrs[ind])

    ax.set_xticks([])
    ax.set_yticks([])

    ax.axis('square')
    ax.axis([-1, sze+1, -1, sze+1])#xmin=0 - 0.8, xmax=sze + 0.8, ymin=0 - 0.8, ymax= sze+ 0.8)

    fig1.tight_layout()
    # fig.show()#savefig("../figs/detections+cam.png")

    return fig0, fig1, detections

def make_plot_sub( n_inds = 10,
                   sze = 5,
                   speed = 0.1,
                   tsteps = 1000,
                   cam_fprint = 1,
                   cam_step = 0.3,
                   y_steps = 2):

    df = make_positions(n_inds, speed, sze, tsteps)
    camx, camy = mow_the_lawn_2D(0, 0, sze, sze, cam_step, y_steps)

    # generate detections (i.e. where individual is in a camera frame)

    detections = get_in_camera_dets(df, camx, camy, cam_fprint)

    # make plots

    # get number of timesteps in camera track
    n_tsteps = len(camx)


    markersize = 20
    clrs =  plt.rcParams['axes.prop_cycle'].by_key()['color']


    fig, ax = plt.subplots(1,2)
    # plot just animal positions
    stp = 10
    for i in df["individual"]:
        print(i)
        ax[0].scatter(df.loc[i][["x_{}".format(i) for i in range(1, n_tsteps+1, stp)]].to_numpy(),
                  df.loc[i][["y_{}".format(i) for i in range(1, n_tsteps+1, stp)]].to_numpy(), marker='.', s=markersize
                      , c=clrs[i])
        ax[0].scatter(df.loc[i]["x_0"], df.loc[i]["y_0"], marker='+', color='black',s=100)

    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # add camera path
    ax[0].plot(camx, camy)
    ax[0].axis('square')
    ax[0].axis([-1, sze + 1, -1,sze + 1])


    for t in detections["time"].unique():
        tdf = detections[detections["time"] == t]

        ax[1].add_patch(Rectangle(
            (tdf["camera_x"].to_numpy()[0] - 0.5 * cam_fprint, tdf["camera_y"].to_numpy()[0] - 0.5 * cam_fprint),
            cam_fprint, cam_fprint,
            facecolor=(0.1, 0, .8, 0.05),
            edgecolor=(0.1, 0, .8, 0.5)))

    for ind in range(n_inds):#, ind in enumerate(detections["individual"].unique()):
        # print("2", ind)

        idf = detections[detections["individual"] == ind]
        if len(idf) > 0:
            ax[1].scatter(idf["x"], idf[["y"]], marker='.', s=markersize, c=clrs[ind])

    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[1].axis('square')
    ax[1].axis([-1, sze+1, -1, sze+1])#xmin=0 - 0.8, xmax=sze + 0.8, ymin=0 - 0.8, ymax= sze+ 0.8)

    fig.tight_layout()
    # fig.show()#savefig("../figs/detections+cam.png")

    return fig


if __name__ == "__main__":
    # # make animal positions
    # n_inds = 10
    # sze = 5
    # speed = 0.1
    # tsteps = 1000
    #
    #
    #
    #
    # # make camera positions
    # cam_fprint = 1
    # cam_step = 0.3
    # y_steps = 2

    fig0, fig1, dets = make_plots()
    fig0.show()
    fig1.show()

    inp = input("save figures?")
    if inp.lower() == 'y':
        print("Saved figures and detections to ../figs/ ")
        fig0.savefig("../figs/positions+track.png")
        fig1.savefig("../figs/detections+cam.png")
        dets.to_csv("../figs/detections.csv")

    a=10