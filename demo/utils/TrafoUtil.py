import numpy as np


""" COORDINATE TRANSFORMATIONS. """
def cam_id(kinect_id, frame):
    """ Returns the index in the calib[mat] list. """
    assert (frame == 'd') or (frame == 'c'), "Frame has to be c(-olor) or d(-epth)."
    i = 0
    if frame == 'c':
        i = 1
    return kinect_id*2 + i


def to_hom(coords):
    """ Turns the [N, D] coord matrix into homogeneous coordinates [N, D+1]. """
    coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], 1)
    return coords_h


def from_hom(coords_h):
    """ Turns the homogeneous coordinates [N, D+1] into [N, D]. """
    coords = coords_h[:, :-1] / (coords_h[:, -1:] + 1e-10)
    return coords


def correct_world(coord3d, calib_data):
    """ Not sure. """
    cam_e_c = np.array(calib_data[1][1])
    coord3d_trafo = from_hom(np.matmul(to_hom(coord3d), np.transpose(np.linalg.inv(cam_e_c))))
    return coord3d_trafo


def trafo_cam2cam(coord3d, camidFrom, camidTo, calib_data):
    """ Trafo 3d coords from one cam frame to another one. """
    coord3d_w = trafo_cam2world(coord3d, camidFrom, calib_data)
    coord3d_cam = trafo_world2cam(coord3d_w, camidTo, calib_data)
    return coord3d_cam


def trafo_world2cam(coord3d, camid, calib_data):
    """ Trafo 3d coords from world frame to some camera frame. """
    cam_e = np.array(calib_data[camid][1])
    coord3d_trafo = from_hom(np.matmul(to_hom(coord3d), np.transpose(cam_e)))
    return coord3d_trafo


def trafo_cam2world(coord3d, camid, calib_data):
    """ Trafo 3d coords from some camera frame to the world frame. """
    cam_e = np.array(calib_data[camid][1])
    coord3d_trafo = from_hom(np.matmul(to_hom(coord3d), np.transpose(np.linalg.inv(cam_e))))
    return coord3d_trafo


def project_into_cam(coord3d, camid, calib_data):
    """ Projection onto the image plane. """
    cam_i = np.array(calib_data[camid][0])
    coord2d = from_hom(np.matmul(coord3d, np.transpose(cam_i)))
    return coord2d


def project_from_world_to_view(coord3d, camid, calib_data):
    """ Trafo from world frame to some camera frame and then project into it. """
    coord3d_trafo = trafo_world2cam(coord3d, camid, calib_data)
    coord2d = project_into_cam(coord3d_trafo, camid, calib_data)
    return coord2d


""" DATA CONVERSION. """
def depth_uint2float(depth_map):
    upper = depth_map[:, :, 0].astype(np.float32) * 256
    lower = depth_map[:, :, 1].astype(np.float32)
    return upper + lower


def depth_float2uint(depth_map):
    depth_map = np.round(depth_map)
    lower = (depth_map % 256).astype(np.uint8)
    upper = np.floor(depth_map / 256).astype(np.uint8)
    return upper, lower


def infrared_scale(infrared_raw, infrared_limits=None):
    assert len(infrared_raw.shape) == 3, "Needs to be MxNx3 image"

    # cast to a single values map
    infrared = depth_uint2float(infrared_raw)

    # scale from 0 .. 1
    min_v, max_v = np.min(infrared), np.max(infrared)
    infrared = (infrared - min_v) / (max_v - min_v)

    # correct dynamics (as done by SDK)
    infrared = np.power(infrared, 0.32)

    # cast to uint8
    if infrared_limits is None:
        infrared = (infrared * 255).astype(np.uint8)
    else:
        assert infrared_limits[0]<infrared_limits[1], "Min has to be smaller than max of infrared_limits"
        infrared = np.clip(infrared * 255, infrared_limits[0], infrared_limits[1])
        min_v, max_v = np.min(infrared), np.max(infrared)
        infrared = ((infrared - min_v) / (max_v - min_v) * 255).astype('uint8')

    return infrared


def kinect2coco(coords_kin, vis_kin):
    """ Converts from the Kinect SDK keypoint definition to coco. """
    # this maps from coco to kinect
    id_dict = {0: 3, 1: 2,  # face, neck
               2: 8, 3: 9, 4: 10,  # right arm
               5: 4, 6: 5, 7: 6,  # left arm
               8: 16, 9: 17, 10: 18,  # right leg
               11: 12, 12: 13, 13: 14}  # left leg

    coord_coco = np.zeros((18, 3))
    coord_vis = np.zeros((18, ))

    for kp_id in range(14):
        kp_id_kinect = id_dict[kp_id]
        coord_coco[kp_id, :] = coords_kin[kp_id_kinect, :]
        coord_vis[kp_id] = vis_kin[kp_id_kinect]
    return coord_coco, coord_vis == 1.0


def tome2coco(coords_tome, vis_tome):
    """ Converts from Tomes keypoint definition to coco. """
    # this maps from coco to tome
    id_dict = {0: 9, 1: 8,  # face, neck
               2: 14, 3: 15, 4: 16,  # right arm
               5: 11, 6: 12, 7: 13,  # left arm
               8: 1, 9: 2, 10: 3,  # right leg
               11: 4, 12: 5, 13: 6}  # left leg

    coord_coco = np.zeros((18, 3))
    coord_vis = np.zeros((18, ))

    for kp_id in range(14):
        kp_id_tome = id_dict[kp_id]
        coord_coco[kp_id, :] = coords_tome[kp_id_tome, :]
        coord_vis[kp_id] = vis_tome[kp_id_tome]
    return coord_coco, coord_vis == 1.0


