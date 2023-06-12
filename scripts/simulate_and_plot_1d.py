import pandas as pd
import numpy as np


def move_fauna_1D_norm(faunaX, speed_var):
    XY = faunaX
    XY += np.random.normal(0, speed_var, size=np.shape(XY))
    return XY

def make_positions(num_inds, speed_var, size, timesteps):

    df = pd.DataFrame()
    df["individual"] = range(num_inds)
    # df["t"] = np.zeros(num_inds).astype(int)
    x= initial_positions(num_inds, size)
    df["x_0"] = x


    for i in range(timesteps):
        tstep = i + 1
        tdf = pd.DataFrame()
        #     tdf["individual"] = range(num_inds)
        #     tdf["t"] = np.zeros(num_inds).astype(int)+ts
        x = move_fauna_1D_norm(x,speed_var)
        tdf["x_{}".format(tstep)] = x

        df = pd.concat([df, tdf], axis=1)

    return df



def initial_positions(num_inds, area):
    '''
    :param num_inds: number of individuals to simulate
    :param area: boundaries of survey area [xmin, ymin, xmax, ymax]
    :return: xy starting positions of indivudals
    '''

    fx = np.random.uniform(0, area, num_inds)

    return fx


class ConfigSim:
    def __init__(self,
                 survey_width,
                 n_fauna,
                 fauna_speed,
                 camera_footprint,
                 camera_step,

                 bg_prob):
        self.survey_width = survey_width
        self.n_fauna = n_fauna
        self.fauna_speed = fauna_speed
        self.camera_footprint = camera_footprint
        self.camera_step = camera_step
        self.background_probability = bg_prob

if __name__ == "__main__":
    area = 5
    n_fauna = 2
    speed = 0.05
    cam_footprint = 1
    cam_step = 0.5



    df = make_positions(5, 0.1, 10, 10)
    print(df.head())