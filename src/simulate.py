import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def initial_positions(num_inds, area):
    '''
    :param num_inds: number of individuals to simulate
    :param area: boundaries of survey area [xmin, ymin, xmax, ymax]
    :return: xy starting positions of indivudals
    '''

    fx = np.random.uniform(area[0], area[1], num_inds)
    fy = np.random.uniform(area[2], area[3], num_inds)

    return fx, fy

def move_fauna_2D_norm(faunaX, faunaY, speed_var):
    XY = np.c_[faunaX, faunaY]
    XY += np.random.normal(0, speed_var, size=np.shape(XY))
    return XY[:, 0], XY[:, 1]

def make_detections(num_inds, speed_var, size, timesteps):
    '''
    :param num_inds: number of individuals to simulate
    :param speed_var: variance of distribution of pos at t+1
    :param area: boundaries of survey area [xmin, xmax, ymin, ymax]
    :param timesteps: number of timesteps to simulate
    :return: positions of each individual
    '''

    area = [0, size, 0, size]
    df = pd.DataFrame()

    df["individual"] = range(num_inds)
    # df["t"] = np.zeros(num_inds).astype(int)
    x, y = initial_positions(num_inds, area)
    df["x_0"] = x
    df["y_0"] = y

    for i in range(timesteps):
        tstep = i + 1
        tdf = pd.DataFrame()
        #     tdf["individual"] = range(num_inds)
        #     tdf["t"] = np.zeros(num_inds).astype(int)+ts
        x, y = move_fauna_2D_norm(x, y, speed_var)
        tdf["x_{}".format(tstep)] = x
        tdf["y_{}".format(tstep)] = y
        df = pd.concat([df, tdf], axis=1)

    return df

def plot_detections(dets, size, timesteps, stp=1):
    fig, ax = plt.subplots(figsize=(10,10))

    for i in dets["individual"]:
        ax.scatter(dets.loc[i][["x_{}".format(i) for i in range(1, timesteps+1, stp)]].to_numpy(),
                  dets.loc[i][["y_{}".format(i) for i in range(1, timesteps+1, stp)]].to_numpy(), marker='.', s=10)
        ax.scatter(dets.loc[i]["x_0"], dets.loc[i]["y_0"], marker='+', color='black',s=100)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.axis(xmin=0-0.5, xmax=size+.5, ymin=0-.5, ymax=size+.5)
    ax.set_aspect('equal')

    return fig, ax


def plot_cam_dets(dets, size, footprint, cam_step, plot_less=0, y_steps=2, stp=10):
    x, y = mow_the_lawn_2D(0, 0, size, size, cam_step, y_steps)
    if plot_less==0:

        # x, y = mow_the_lawn_2D(0, 0, size, size, cam_step, y_steps)
        fig, ax = plot_detections(dets, size, len(x), stp=stp)
        ax.plot(x, y)
        ax.axis('equal')

        ax.add_patch(Rectangle((0 - 0.5 * footprint, 0 - 0.5 * footprint), footprint, footprint, alpha=0.6))
        ax.axis(xmin=0 - 0.8, xmax=size + .8, ymin=0 - .8, ymax=size + 0.8)

    else:

        fig, ax = plot_detections(dets, size, plot_less, stp=stp)
        ax.plot(x, y)
        ax.axis('equal')

        ax.add_patch(Rectangle((0 - 0.5 * footprint, 0 - 0.5 * footprint), footprint, footprint, alpha=0.6))
        ax.axis(xmin=0 - 0.8, xmax=size + .8, ymin=0 - .8, ymax=size + 0.8)

    return fig, ax, x,y


def mow_the_lawn_2D(startx, starty, endx, endy, camstep, ysteps):
    # first leg, x increasing to max
    x = startx
    y = starty

    pathX = [startx]
    pathY = [starty]

    while y < endy:
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


def zamboni2D(startx, starty, endx, endy, camstep, ysteps):
    x = startx
    y = starty

    pathX = [startx]
    pathY = [starty]

    ylength = (endy - starty + ysteps * camstep) / 2

    posysteps = ylength / camstep
    negysteps = (ylength - ysteps * camstep) / camstep
    topy = starty
    while topy < endy:
        while x < endx:
            # take step in +ve x direction
            x += camstep
            # add new position to
            pathX.append(x)
            pathY.append(y)

        # go in positive y for posysteps
        i = 0
        while i < posysteps:
            y += camstep

            pathX.append(x)
            pathY.append(y)
            i += 1
        topy = y
        # go back to startx
        while x > startx:
            x -= camstep
            pathX.append(x)
            pathY.append(y)

        # take negysteps steps in -y dir
        i = 0
        while i < negysteps:
            y -= camstep
            pathX.append(x)
            pathY.append(y)
            i += 1

    return pathX, pathY


def in_footprint_2D(camx, camy, footprint, faunax, faunay):
    sp = np.logical_and(np.logical_and(faunax >= (camx - (footprint / 2)), faunax <= (camx + (footprint / 2))),
                        np.logical_and(faunay >= (camy - (footprint / 2)), faunay <= (camy + (footprint / 2))))
    return sp


def cam2D(footprint, camx, camy, faunax, faunay, faunastep):
    detections = []
    t = 0
    idx = 0

    # step through each camera position
    for i in range(len(camx)):
        x = camx[i]
        y = camy[i]
        # check if any fauna are in the camera footprint
        sp = in_footprint_2D(x, y, footprint, faunax, faunay)

        if any(sp):
            # spotted = np.where(sp)
            fx = faunax[np.where(sp)]
            fy = faunay[np.where(sp)]
            for j in range(len(fx)):  # f in fx:
                detection = {'id': idx,
                             'camera_location_x': x,
                             'camera_location_y': y,
                             'time': t,
                             'object_location_x': fx[j],
                             'object_location_y': fy[j]}
                detections.append(detection)
                idx += 1
        faunax, faunay = move_fauna_2D_norm(faunax, faunay, faunastep)


        t += 1

    return detections
