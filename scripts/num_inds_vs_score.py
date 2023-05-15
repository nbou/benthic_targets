from src.simulate import make_positions, mow_the_lawn_2D
import pandas as pd
from src.hypotheses import build_and_score_hypotheses_multi,generate_probability_matrix_2D

import math
import numpy as np
if __name__ == "__main__":
    df = pd.read_csv('../figs/detections.csv')
    df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    print(df.head())

    speed = 0.1
    # dists = generate_probability_matrix_2D(df, speed)
    # # dists = np.zeros((len(df), len(df)))
    # # for i in range(len(df)):
    # #     for j in range(len(df)):
    # #         xi = df.loc[i]["x"]
    # #         xj = df.loc[j]["x"]
    # #         yi = df.loc[i]["y"]
    # #         yj = df.loc[j]["y"]
    # #         dists[i, j] = np.linalg.norm(np.array((xi, yi)) - np.array((xj, yj)))
    #
    # print(dists)

    hyps, scores = build_and_score_hypotheses_multi(df[:20], speed, 10/25, d2=True)

    a = 10
