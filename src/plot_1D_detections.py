import numpy as np
# import utils
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def move_fauna_1D(faunax, maxstep):
    newFaunax = faunax + np.random.uniform(-maxstep,maxstep, np.shape(faunax))
    return newFaunax

def move_fauna_1D_norm(faunax, maxstep):
    newFaunax = faunax + np.random.normal(0, maxstep, np.shape(faunax)) #np.random.uniform(-maxstep,maxstep, np.shape(faunax))
    return newFaunax

# xmin=0
# xmax=5
# fauna_step=0.2
# nfauna=5
# cam_init_x=-1
# cam_init_y=0
# cam_fp=1.25
# cam_step=1
# fauna = np.array([0.5, 0.75, 2, 2.5, 4.25])#np.random.uniform(np.random.uniform(xmin, xmax, nfauna))

xmin=0
xmax=3
fauna_step=0.2
nfauna=5
cam_init_x=-1
cam_init_y=0
cam_fp=1.1
cam_step=1
fauna = np.array([0.25,1.25,2.25])


cam = np.arange(xmin,xmax,cam_step)#np.array([0, 0.5, 1, 1.5, 2])


fig, ax = plt.subplots()
for i in range(len(cam)):

    camx = cam[i]
    camy = i
    ax.scatter(fauna, np.ones(np.shape(fauna))*i)
    ax.add_patch(patches.Rectangle(
        (camx - cam_fp / 2, camy - cam_fp / 4),
        cam_fp,
        cam_fp/2,
        alpha=0.3))
    print(fauna)
    fauna = move_fauna_1D_norm(fauna,fauna_step)
ax.set_ylabel('Time step')
ax.set_yticks([0,1,2])
# ax.set_xticks([])
# ax.set_xticklabels([])
plt.show()