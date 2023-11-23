import numpy as np
import matplotlib.cm


def draw_person_limbs_2d_coco(axis, coords, vis=None, color=None, order='hw', with_face=True):
    """ Draws a 2d person stick figure in a matplotlib axis. """
    import matplotlib.cm
    if order == 'uv':
        pass
    elif order == 'hw':
        coords = coords[:, ::-1]
    else:
        assert 0, "Unknown order."

    LIMBS_COCO = np.array([[1, 2], [2, 3], [3, 4],  # right arm
                           [1, 8], [8, 9], [9, 10],  # right leg
                           [1, 5], [5, 6], [6, 7],  # left arm
                           [1, 11], [11, 12], [12, 13],  # left leg
                           [1, 0], [2, 16], [0, 14], [14, 16], [0, 15], [15, 17], [5, 17]])  # head

    if type(color) == str:
        if color == 'sides':
            blue_c = np.array([[0.0, 0.0, 1.0]])  # side agnostic
            red_c = np.array([[1.0, 0.0, 0.0]])  # "left"
            green_c = np.array([[0.0, 1.0, 0.0]])  # "right"
            color = np.concatenate([np.tile(green_c, [6, 1]),
                                    np.tile(red_c, [6, 1]),
                                    np.tile(blue_c, [7, 1])], 0)
            if not with_face:
                color = color[:13, :]

    if not with_face:
        LIMBS_COCO = LIMBS_COCO[:13, :]

    if vis is None:
        vis = np.ones_like(coords[:, 0]) == 1.0

    if color is None:
        color = matplotlib.cm.jet(np.linspace(0, 1, LIMBS_COCO.shape[0]))[:, :3]

    for lid, (p0, p1) in enumerate(LIMBS_COCO):
        if (vis[p0] == 1.0) and (vis[p1] == 1.0):
            if type(color) == str:
                axis.plot(coords[[p0, p1], 0], coords[[p0, p1], 1], color, linewidth=2)
            else:
                axis.plot(coords[[p0, p1], 0], coords[[p0, p1], 1], color=color[lid, :], linewidth=2)


def draw_person_limbs_3d_coco(axis, coords, bones ,vis=None, color=None, azim=-90, elev=-90, with_face=False, rescale=True, linewidth=2):
    """ Draws a 3d person stick figure in a matplotlib axis. """

    import matplotlib.cm

    # LIMBS_COCO = np.array([[1, 2], [2, 3], # head?
    #                        [20, 8], [8, 9], [9, 10], # right arm
    #                        [20, 4], [4, 5], [5, 6], # left arm
    #                        [20, 16], [16, 17], [17, 18], [18,19], 
    #                        # [1, 0],
    #                        [20, 12], [12, 13], [13, 14], [14,15]])#,

                           

    # LIMBS_COCO = np.array([[1, 2], [2, 3], [3, 4],
    #                        [1, 8], [8, 9], [9, 10],
    #                        [1, 5], [5, 6], [6, 7],
    #                        [1, 11], [11, 12], [12, 13],
    #                        [1, 0],
    #                        [0, 14], [14, 16], [0, 15], [15, 17]])#,
                           # [2, 16], [5, 17]])

    LIMBS_COCO = bones


    if not with_face:
        LIMBS_COCO = LIMBS_COCO[:13, :]

    if vis is None:
        vis = np.ones_like(coords[:, 0]) == 1.0

    vis = vis == 1.0

    if type(color) == str:
        if color == 'sides':
            blue_c = np.array([[0.0, 0.0, 1.0]])  # side agnostic
            red_c = np.array([[1.0, 0.0, 0.0]])  # "left"
            green_c = np.array([[0.0, 1.0, 0.0]])  # "right"
            color = np.concatenate([np.tile(green_c, [6, 1]),
                                    np.tile(red_c, [6, 1]),
                                    np.tile(blue_c, [7, 1])], 0)

    if color is None:
        color = matplotlib.cm.jet(np.linspace(0, 1, LIMBS_COCO.shape[0]))[:, :3]

    for lid, (p0, p1) in enumerate(LIMBS_COCO):
        if (vis[p0] == 1.0) and (vis[p1] == 1.0):
            if type(color) == str:
                axis.plot(coords[[p0, p1], 0], coords[[p0, p1], 1], coords[[p0, p1], 2], color, linewidth=linewidth)
            else:
                axis.plot(coords[[p0, p1], 0], coords[[p0, p1], 1], coords[[p0, p1], 2], color=color[lid, :], linewidth=linewidth)

    if np.sum(vis) > 0 and rescale:
        min_v, max_v, mean_v = np.min(coords[vis, :], 0), np.max(coords[vis, :], 0), np.mean(coords[vis, :], 0)
        range = np.max(np.maximum(np.abs(max_v-mean_v), np.abs(mean_v-min_v)))
        axis.set_xlim([-2, 2])
        axis.set_ylim([-2, 2])
        axis.set_zlim([-2, 2])

    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_zlabel('z')
    axis.view_init(azim=azim, elev=elev)


def save_3d_pred(save_name,
                 coords_p, vis_p,
                 coords_gt, vis_gt,
                 # coords_cmp=None, vis_cmp=None,
                 draw_cam=True, view_id=0):
    """ Creates a matplotlib figure and plots the given normalized relative coordinates, adds a camera and saves as png."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    coords_p = np.copy(coords_p)
    coords_gt = np.copy(coords_gt)
    # vis_p = vis_p[:14, ]
    # coords_p = coords_p[:14, :]

    right_inds = [5, 6, 7, 11, 12, 13]
    # right_inds = [7, 13]
    m = np.ones_like(vis_p) == 0.0
    m[right_inds, ] = True
    coords_p = np.stack([coords_p[:, 0], coords_p[:, 2], -coords_p[:, 1]], -1)

    fig = plt.figure(1)
    ax2 = fig.add_subplot('111', projection='3d')
    draw_person_limbs_3d_coco(ax2, coords_p, vis_p, color='sides', rescale=False, linewidth=6, with_face=False)
    vis_p = np.logical_and(vis_p == 1.0, m)
    # ax2.plot(coords_p[vis_p, 0], coords_p[vis_p, 1], coords_p[vis_p, 2], 'go')

    if coords_gt is not None:
        coords_gt = np.stack([coords_gt[:, 0], coords_gt[:, 2], -coords_gt[:, 1]], -1)
        draw_person_limbs_3d_coco(ax2, coords_gt, vis_gt, color='b--', rescale=False, linewidth=6, with_face=False)
        vis_gt = np.logical_and(vis_gt == 1.0, m)
        # ax2.plot(coords_gt[vis_gt, 0], coords_gt[vis_gt, 1], coords_gt[vis_gt, 2], 'bo')

    # if coords_cmp is not None:
    #     coords_cmp = np.stack([coords_cmp[:, 0], coords_cmp[:, 2], -coords_cmp[:, 1]], -1)
    #     draw_person_limbs_3d_coco(ax2, coords_cmp, vis_cmp, color='b:', rescale=False, linewidth=3)
    #     vis_cmp = np.logical_and(vis_cmp == 1.0, m)
    #     ax2.plot(coords_cmp[vis_cmp, 0], coords_cmp[vis_cmp, 1], coords_cmp[vis_cmp, 2], 'bo')
    if view_id == 0:
        ax2.view_init(azim=-45., elev=30.)
    elif view_id == 1:
        ax2.view_init(azim=-90., elev=40.)
    elif view_id == 2:
        ax2.view_init(azim=0., elev=0.)
    else:
        ax2.view_init(azim=140., elev=50.)

    # size = 1.0
    # ax2.set_xlim([-size, size])
    # y_mean = np.mean(coords_gt[vis_gt == 1.0, 1])
    # print('y_mean', y_mean)
    # cam_dist = y_mean-size
    # ax2.set_ylim([0, 3])
    # ax2.set_ylim([0.5, 5.0])
    # ax2.set_zlim([-1, 0.75])

    min_v, max_v, mean_v = np.min(coords_gt[vis_gt, :], 0), np.max(coords_gt[vis_gt, :], 0), np.mean(coords_gt[vis_gt, :], 0)
    range = np.max(np.maximum(np.abs(max_v-mean_v), np.abs(mean_v-min_v)))
    ax2.set_xlim([mean_v[0]-range, mean_v[0]+range])
    ax2.set_ylim([mean_v[1]-range, mean_v[1]+range])
    ax2.set_zlim([mean_v[2]-range, mean_v[2]+range])
    cam_dist = - mean_v[1] + range

    if draw_cam:
        # cam_dist = 0.0
        vp_size = 0.25
        cone_length = 0.75
        ax2.plot([vp_size, vp_size, -vp_size, -vp_size, vp_size, vp_size],
                 [-cam_dist, -cam_dist, -cam_dist, -cam_dist, -cam_dist, -cam_dist],
                 [vp_size, -vp_size, -vp_size, vp_size, vp_size, -vp_size], 'b', linewidth=4)
        ax2.plot([vp_size, 0.0, -vp_size, 0.0, vp_size, 0.0, -vp_size, 0.0],
                 [-cam_dist, -(cam_dist+cone_length), -cam_dist, -(cam_dist+cone_length),
                  -cam_dist, -(cam_dist+cone_length), -cam_dist, -(cam_dist+cone_length)],
                 [vp_size, 0.0, vp_size, 0.0, -vp_size, 0.0, -vp_size, 0.0], 'b', linewidth=4)

    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_zlabel('')
    # plt.show()
    fig.savefig(save_name)
    plt.close(fig)