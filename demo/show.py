"""
    This script shows how to iterate the dataset and how to use the data stored in the json files.
    It will show color image, depth map and infrared map and project the 3D annotation into the 2D views.
 """
import json
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from mpl_toolkits.mplot3d import Axes3D


from utils.TrafoUtil import *
from utils.DrawUtils import *

# decide which data to show
PATH_TO_DATA = './data/training/'  # MKV train
#PATH_TO_DATA = './data/validation/'  #MKV val

# PATH_TO_DATA = './data/captury_train/'  # CAP train
# PATH_TO_DATA = './data/captury_test/'  # CAP val
# PATH_TO_DATA = './data/captury_test_obj_non_frontal/'  # CAP val_ss

# load skeleton annotations
with open(PATH_TO_DATA + 'anno.json', 'r') as fi:
    anno = json.load(fi)
print('Loaded a total of %d frames.' % len(anno))

# load Kinect SDK predictions
with open(PATH_TO_DATA + 'pred_sdk.json', 'r') as fi:
    pred_sdk = json.load(fi)
print('Loaded a total of %d frames.' % len(pred_sdk))

# load camera matrices
with open(PATH_TO_DATA + 'calib.json', 'r') as fi:
    calib = json.load(fi)

# load info
with open(PATH_TO_DATA + 'info.json', 'r') as fi:
    info = json.load(fi)

# 3D joint visualization
# Iterate through the frames and visualize the predicted 3D poses
for fid, preds in enumerate(pred_sdk):
    print(f'Visualizing frame {fid}')
    
    for pid, person_pred in enumerate(preds):
        # person_pred 구조에서 실제 3D 좌표가 들어있는 첫 번째 리스트를 추출합니다.
        # 여기서는 가시성 정보를 포함하지 않은 실제 좌표만 추출합니다.
        coords3d_pred = [np.array(coords) for coords in person_pred[0][0] if len(coords) == 3]
        
        if coords3d_pred:
            coords3d_pred = np.array(coords3d_pred)
            
            # Create a 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot for joints
            scatter = ax.scatter(coords3d_pred[:, 0], coords3d_pred[:, 1], coords3d_pred[:, 2], s=60)

            # Draw limbs
            draw_person_limbs_3d_coco(ax, coords3d_pred, color='blue', linewidth=2)

            # Annotate each point with a number
            for i, txt in enumerate(range(coords3d_pred.shape[0])):
                ax.text(coords3d_pred[i, 0], coords3d_pred[i, 1], coords3d_pred[i, 2], '%d' % txt, size=20, zorder=1, color='k')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title(f'Frame {fid} - Person {pid} 3D Pose')
            plt.show()
        
        else:
            print(f"No 3D coordinates to visualize for frame {fid}, person {pid}")

# iterate frames
for fid, frame_anno in enumerate(anno):
    print('Frame %d: %d person(s)' % (fid, len(frame_anno)))
    print('Location %d' % info['location'][fid])
    print('Subject', info['subject'][fid])

    calib_frame = list()
    for cid in calib['map'][fid]:
        calib_frame.append(calib['mat'][cid])

    # iterate individual persons found
    for person_anno in frame_anno:
        
        coord3d = np.array(person_anno['coord3d'])  # Save already as modified
        print(coord3d.shape)
        vis = np.array(person_anno['vis'])
        i = 0
        # show in available cameras
        num_kinect = len(calib_frame) // 2  # because each kinect has a depth and a color frame
        for kid in range(num_kinect):
            color_cam_id, depth_cam_id = cam_id(kid, 'c'), cam_id(kid, 'd')
            
            img_path = PATH_TO_DATA + 'color/' + '%d_%08d.png' % (color_cam_id, fid)
            img = Image.open(img_path)
            img = np.array(img)
            
            depth_path = PATH_TO_DATA + 'depth/' + '%d_%08d.png' % (depth_cam_id, fid)
            depth = depth_uint2float(imageio.imread(depth_path))
            
            infrared_path = PATH_TO_DATA + 'infrared/' + '%d_%08d.png' % (depth_cam_id, fid)
            infrared = imageio.imread(infrared_path)
            
            if len(infrared.shape) == 2:
                infrared = np.stack([infrared, infrared, infrared], -1)
            infrared = infrared_scale(infrared, [30, 150])

            # project 3d coordinates into view
            coord2d_c = project_from_world_to_view(coord3d, color_cam_id, calib_frame)
            coord2d_d = project_from_world_to_view(coord3d, depth_cam_id, calib_frame)
            print(coord2d_c.shape)

            # show data
            with_face = None
            print("print GT")
            fig = plt.figure()
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            ax1.imshow(img)
            draw_person_limbs_2d_coco(ax1, coord2d_c, vis, order='uv', with_face=with_face)
            ax2.imshow(depth)
            draw_person_limbs_2d_coco(ax2, coord2d_d, vis, color='g', order='uv', with_face=with_face)
            ax3.imshow(infrared)
            draw_person_limbs_2d_coco(ax3, coord2d_d, vis, order='uv', with_face=with_face)
            
            
            # for coords3d_sdk_pred, vis_sdk in zip(pred_sdk[fid][kid][0], pred_sdk[fid][kid][1]):
            #     coord2d_sdk_c = project_from_world_to_view(np.array(coords3d_sdk_pred), depth_cam_id, calib_frame)
            #     draw_person_limbs_2d_coco(ax2, coord2d_sdk_c, vis_sdk, color='r', order='uv', with_face=False)
                
            plt.show()
