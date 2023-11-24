# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.append(".")

from argparse import ArgumentParser
from typing import Dict
import json
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.TrafoUtil import *
from utils.DrawUtils import *
from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs',
        type=str,
        nargs='?',
        default = './tests/data/kinect/color/1_00000003.png',
        help='Input image/video path or folder path.')
    parser.add_argument(
        '--pose2d',
        type=str,
        default=None,
        help='Pretrained 2D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose2d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose2d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument(
        '--pose3d',
        type=str,
        default='human3d',
        help='Pretrained 3D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose3d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose3d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument(
        '--det-model',
        type=str,
        default=None,
        help='Config path or alias of detection model.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the checkpoints of detection model.')
    parser.add_argument(
        '--det-cat-ids',
        type=int,
        nargs='+',
        default=0,
        help='Category id for detection model.')
    parser.add_argument(
        '--scope',
        type=str,
        default='mmpose',
        help='Scope where modules are defined.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image/video in a popup window.')
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        help='Whether to draw the bounding boxes.')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Whether to draw the predicted heatmaps.')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--use-oks-tracking',
        action='store_true',
        help='Whether to use OKS as similarity in tracking')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset, and move the bbox (along with the '
        '2D pose) to the average bbox center of the dataset. This is useful '
        'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization.')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization.')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--black-background',
        action='store_true',
        help='Plot predictions on a black image')
    parser.add_argument(
        '--vis-out-dir',
        type=str,
        default='./vis_results/human3d',
        help='Directory for saving visualized results.')
    parser.add_argument(
        '--pred-out-dir',
        type=str,
        default='./vis_results/human3d',
        help='Directory for saving inference results.')
    parser.add_argument(
        '--show-alias',
        action='store_true',
        help='Display all the available model aliases.')

    call_args = vars(parser.parse_args())

    init_kws = [
        'pose2d', 'pose2d_weights', 'scope', 'device', 'det_model',
        'det_weights', 'det_cat_ids', 'pose3d', 'pose3d_weights'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    diaplay_alias = call_args.pop('show_alias')

    return init_args, call_args, diaplay_alias


# MMPose와 Kinect SDK 간의 조인트 매핑 테이블
joint_mapping = {
    'PELVIS': {'mmpose': 0, 'kinect': 0},
    'SPINE_NAVAL': {'mmpose': 7, 'kinect': 1},
    'SPINE_CHEST': {'mmpose': 8, 'kinect': 20},
    'NECK': {'mmpose': 9, 'kinect': 2},
    'SHOULDER_LEFT': {'mmpose': 14, 'kinect': 4},
    'ELBOW_LEFT': {'mmpose': 15, 'kinect': 5},
    'WRIST_LEFT': {'mmpose': 16, 'kinect': 6},
    'SHOULDER_RIGHT': {'mmpose': 11, 'kinect': 8},
    'ELBOW_RIGHT': {'mmpose': 12, 'kinect': 9},
    'WRIST_RIGHT': {'mmpose': 13, 'kinect': 10},
    'HIP_LEFT': {'mmpose': 1, 'kinect': 12},
    'KNEE_LEFT': {'mmpose': 2, 'kinect': 13},
    'ANKLE_LEFT': {'mmpose': 3, 'kinect': 14},
    'HIP_RIGHT	': {'mmpose': 4, 'kinect': 16},
    'KNEE_RIGHT': {'mmpose': 5, 'kinect': 17},
    'ANKLE_RIGHT': {'mmpose': 16, 'kinect': 18},
    'HEAD': {'mmpose': 10, 'kinect': 3}
}

LIMBS_COCO = np.array([[1, 2], [2, 3], [3, 4],  # right arm
                       [1, 8], [8, 9], [9, 10],  # right leg
                       [1, 5], [5, 6], [6, 7],  # left arm
                       [1, 11], [11, 12], [12, 13],  # left leg
                       [1, 0], [2, 16], [0, 14], [14, 16], [0, 15], [15, 17], [5, 17]])  # head   

# LIMBS_MMPOSE = np.array([[8, 11], [11, 12], [12, 13],  # right arm
#                          [0, 4], [4, 5], [5, 6],  # right leg
#                          [8, 14], [14, 15], [15, 16],  # left arm
#                          [0, 1], [1, 2], [2, 3],  # left leg
#                          [0, 7], [7, 8], [8, 9], [9, 10]])  # head
LIMBS_MMPOSE = np.array([[8, 11], [11, 12], [12, 13],  # right arm
                         [8, 4], [4, 5], [5, 6],  # right leg
                         [8, 14], [14, 15], [15, 16],  # left arm
                         [8, 1], [1, 2], [2, 3],  # left leg
                         [8, 9], [9, 10]])  # head

# LIMBS_KINECT = np.array([[20, 3], [12, 20], [20, 16], # head
#                          [20, 8], [8, 9], [9, 10], # right arm
#                          [20, 4], [4, 5], [5, 6], # left arm
#                          [16, 17], [17, 18], # right leg
#                          [12, 13], [13, 14]]) # left leg

LIMBS_KINECT = np.array([[20, 2], [2, 3], # head
                         [20, 8], [8, 9], [9, 10], # right arm
                         [20, 4], [4, 5], [5, 6], # left arm
                         [20, 16], [16, 17], [17, 18], # right leg
                         [20, 12], [12, 13], [13, 14]]) # left leg --> 왼쪽 다리 안나옴


def display_model_aliases(model_aliases: Dict[str, str]) -> None:
    """Display the available model aliases and their corresponding model
    names."""
    aliases = list(model_aliases.keys())
    max_alias_length = max(map(len, aliases))
    print(f'{"ALIAS".ljust(max_alias_length+2)}MODEL_NAME')
    for alias in sorted(aliases):
        print(f'{alias.ljust(max_alias_length+2)}{model_aliases[alias]}')

# 이미지 파일명에서 프레임 번호 추출 (예: '1_00000200.png' -> 200)
def extract_frame_id(image_path):
    frame_id_match = re.search(r'(\d+)_(\d+).png', image_path)
    if frame_id_match:
        return int(frame_id_match.group(2))  # 두 번째 그룹(숫자)를 프레임 번호로 사용
    else:
        print("Could not extract frame number from the image path")
        return None
    
# 추가된 함수: Kinect SDK 예측 결과를 로드하는 함수
def load_kinect_sdk_predictions(path_to_sdk_json, frame_id):
    # JSON 파일로부터 Kinect SDK의 예측 결과를 로드
    with open(path_to_sdk_json, 'r') as file:
        pred_sdk = json.load(file)

    # 리스트 내에서 인덱스를 기반으로 예측 결과 반환
    if 0 <= frame_id < len(pred_sdk):
        print(f"Loaded prediction for frame index {frame_id}")
        return pred_sdk[frame_id]
    else:
        print(f"No prediction data found for frame index {frame_id}")
        return None
    
def load_gt(path_to_gt_json, frame_id):
    # JSON 파일로부터 Kinect SDK의 예측 결과를 로드
    with open(path_to_gt_json, 'r') as file:
        anno_gt = json.load(file)

    # 리스트 내에서 인덱스를 기반으로 예측 결과 반환
    if 0 <= frame_id < len(anno_gt):
        print(f"Loaded prediction for frame index {frame_id}")
        return anno_gt[frame_id]
    else:
        print(f"No prediction data found for frame index {frame_id}")
        return None
    
def filter_kinect_joints(kinect_joints, joint_mapping):
    # Kinect SDK 조인트 좌표 배열에서 MMPose와 매핑된 조인트만 추출
    filtered_joints = [kinect_joints[joint_mapping[joint]['kinect']] for joint in joint_mapping]
    return np.array(filtered_joints)

def scale_coords_to_reference(ref_coords, target_coords, joint_mapping):
    # 조인트 매핑 테이블을 사용하여 MMPose와 Kinect SDK의 조인트 인덱스를 얻습니다.
    joint_index_1_ref = joint_mapping['PELVIS']['mmpose']
    joint_index_2_ref = joint_mapping['SPINE_NAVAL']['mmpose']
    joint_index_1_target = joint_mapping['PELVIS']['kinect']
    joint_index_2_target = joint_mapping['SPINE_NAVAL']['kinect']

    # target_coords 배열의 첫 번째 차원을 제거하여 정확한 shape을 얻습니다.
    target_coords = target_coords[0]

    # 배열의 길이 확인
    if len(ref_coords) <= joint_index_2_ref or len(target_coords) <= joint_index_2_target:
        raise ValueError("Index out of bounds. Check joint_mapping and the length of the coordinates arrays.")
    
    # 조인트 사이의 거리 계산
    dist_ref = np.linalg.norm(ref_coords[joint_index_1_ref] - ref_coords[joint_index_2_ref])
    dist_target = np.linalg.norm(target_coords[joint_index_1_target] - target_coords[joint_index_2_target])
    
    # Scale ratio 계산
    if dist_target == 0:
        raise ValueError("Distance between target joints is zero. Can't scale.")
    
    scale_ratio = dist_ref / dist_target
    
    # 모든 좌표에 대한 스케일 조정
    scaled_target_coords = target_coords * scale_ratio

    return scaled_target_coords

def draw_limbs_3d(ax, joints, limbs, color='blue'):
    for limb in limbs:
        i, j = limb
        if i < len(joints) and j < len(joints):  # Check if the joint indices exist
            ax.plot([joints[i, 0], joints[j, 0]],
                    [joints[i, 1], joints[j, 1]],
                    [joints[i, 2], joints[j, 2]], color=color)

# 팔 길이 계산 함수
def calculate_arm_length(shoulder, elbow, wrist):
    upper_arm_length = np.linalg.norm(elbow - shoulder)
    lower_arm_length = np.linalg.norm(wrist - elbow)
    total_arm_length = upper_arm_length + lower_arm_length
    return total_arm_length

def calculate_joint_angles(shoulder, elbow, wrist):
    upper_arm = elbow - shoulder
    forearm = wrist - elbow
    elbow_angle = np.degrees(np.arccos(np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm))))
    body_orientation = np.array([0, 0, 1])
    shoulder_angle = np.degrees(np.arccos(np.dot(upper_arm, body_orientation) / (np.linalg.norm(upper_arm) * np.linalg.norm(body_orientation))))
    return elbow_angle, shoulder_angle

def calculate_arm_data(shoulder_index, elbow_index, wrist_index, coords3d):
    shoulder = coords3d[shoulder_index]
    elbow = coords3d[elbow_index]
    wrist = coords3d[wrist_index]
    elbow_angle, shoulder_angle = calculate_joint_angles(shoulder, elbow, wrist)
    arm_length = calculate_arm_length(shoulder, elbow, wrist)
    return shoulder, elbow, wrist, elbow_angle, shoulder_angle, arm_length

# 팔꿈치 위치 예측 함수
def predict_elbow_position(shoulder, wrist, upper_arm_length):
    # 어깨와 손목 사이의 벡터를 계산합니다.
    arm_vector = wrist - shoulder
    # 벡터를 정규화하여 단위 벡터를 얻습니다.
    arm_direction = arm_vector / np.linalg.norm(arm_vector)
    # 어깨에서 팔꿈치까지의 예상 위치를 계산합니다.
    elbow_position = shoulder + arm_direction * upper_arm_length
    return elbow_position

def calculate_ik_data(coords3d):
    left_data = calculate_arm_data(4, 5, 6, coords3d)
    right_data = calculate_arm_data(8, 9, 10, coords3d)
    
    return {
        "left_arm": {
            "shoulder": left_data[0],
            "elbow": left_data[1],
            "wrist": left_data[2],
            "elbow_angle": left_data[3],
            "shoulder_angle": left_data[4],
            "arm_length": left_data[5]
        },
        "right_arm": {
            "shoulder": right_data[0],
            "elbow": right_data[1],
            "wrist": right_data[2],
            "elbow_angle": right_data[3],
            "shoulder_angle": right_data[4],
            "arm_length": right_data[5]
        }
    }

def reconstruct_arm_joints(shoulder, elbow_angle, shoulder_angle, arm_length):
    upper_arm_length = arm_length / 2
    lower_arm_length = arm_length / 2
    elbow = np.array([shoulder[0] + upper_arm_length * np.sin(np.radians(shoulder_angle)), shoulder[1], shoulder[2] + upper_arm_length * np.cos(np.radians(shoulder_angle))])
    wrist = np.array([elbow[0] + lower_arm_length * np.sin(np.radians(elbow_angle)), elbow[1], elbow[2] + lower_arm_length * np.cos(np.radians(elbow_angle))])
    return np.array([shoulder, elbow, wrist])

def visualize_combined_results(image_path, pred_mmpose, anno_data, pred_sdk, frame_id, vis_out_dir):
    # 이미지 로드
    image = plt.imread(image_path)
   
    # mmpose 3D 포즈 추정 결과 추출
    mmpose_coords3d_pred = np.array(pred_mmpose[0]['predictions'][0][0]['keypoints'])
    print(f"Coords3D MMPose Pred for frame {0}: {mmpose_coords3d_pred}")
    gt_coords3d = np.array(anno_data[0]['coord3d'])

    # Kinect SDK 3D 포즈 예측 결과 추출
    kinect_coords3d_pred = [np.array(coords) for coords in pred_sdk[0][0][0] if len(coords) == 3]
    if not kinect_coords3d_pred:
        print(f"No 3D coordinates to visualize for frame {0}")
        return

    # 1차원 리스트의 리스트를 2차원 NumPy 배열로 변환합니다.
    kinect_coords3d_pred = np.array(kinect_coords3d_pred)

    # 스케일 비율을 사용하여 Kinect SDK 좌표 조정
    # kinect_coords3d_pred_scaled = scale_coords_to_reference(mmpose_coords3d_pred, kinect_coords3d_pred, joint_mapping)
    # no_selected_joints = [0, 1, 2, 7, 11, 15, 19, 21, 22, 23, 24]
    no_selected_joints = [1, 2, 11, 15, 19, 21, 22, 23, 24] # 0, 7 머리 양손 구현 위해  다시 추가
    vis_selected = np.ones(25)
    vis_selected[no_selected_joints] = 0
    # Figure 및 Axes 설정
    # fig, ax = plt.subplots(1, 4, figsize=(15, 5))  # 3개의 서브플롯 생성
    fig = plt.figure(figsize=(20, 10))
    ax = []
    ax.append(fig.add_subplot(1, 4, 1))
    ax.append(fig.add_subplot(1, 4, 2, projection='3d'))
    ax.append(fig.add_subplot(1, 4, 3, projection='3d'))
    ax.append(fig.add_subplot(1, 4, 4, projection='3d'))
    
    # 머리 관절 인덱스 설정
    gt_head_index = 0 # GT 데이터에서 머리 관절의 인덱스
    mmpose_head_index = 10 # MMPose 데이터에서 머리 관절의 인덱스
    kinect_head_index = 3 # Kinect 데이터에서 머리 관절의 인덱스

    # GT 데이터의 머리 관절 좌표로 조정
    gt_head_coord = gt_coords3d[gt_head_index]
    gt_coords3d -= gt_head_coord

    # MMPose 데이터의 머리 관절 좌표로 조정
    mmpose_head_coord = mmpose_coords3d_pred[mmpose_head_index]
    mmpose_coords3d_pred -= mmpose_head_coord

    # Kinect 데이터의 머리 관절 좌표로 조정
    kinect_head_coord = kinect_coords3d_pred[kinect_head_index]
    kinect_coords3d_pred -= kinect_head_coord

    # 첫 번째 서브플롯에 2D 이미지 표시
    # ax[0] = fig.add_subplot(1, 4, 1)
    ax[0].imshow(image)
    ax[0].set_title('2D Image')
    ax[0].axis('off')  # 축 숨기기

    # 두 번째 서브플롯에 3D GT(kinect) 포즈 추정 결과 표시
    # ax[1] = fig.add_subplot(1, 4, 2, projection='3d')
    gt_coords3d = gt_coords3d - gt_coords3d[0] # Rotation 기준점
    gt_coords3d = rotate_joints_y(gt_coords3d, 0)
    ax[1].scatter(gt_coords3d[:13, 0], gt_coords3d[:13, 1], gt_coords3d[:13, 2], s=6)
    draw_person_limbs_3d_coco(ax[1], gt_coords3d, LIMBS_COCO ,color='blue')
    for idx, coord in enumerate(gt_coords3d):
        ax[1].text(coord[0], coord[1], coord[2], f'{idx}', color='black', fontsize=8)
    ax[1].set_title('3D GT')
    ax[1].axis('on')

    # 세 번째 서브플롯에 MMPose 3D 포즈 추정 결과 표시
    # ax[2] = fig.add_subplot(1, 4, 3, projection='3d')
    mmpose_coords3d_pred = mmpose_coords3d_pred - mmpose_coords3d_pred[10] # Rotation 기준점
    mmpose_coords3d_pred = rotate_joints_y(mmpose_coords3d_pred, 180)
    mmpose_coords3d_pred = rotate_joints_z(mmpose_coords3d_pred, 90)
    mmpose_coords3d_pred = rotate_joints_x(mmpose_coords3d_pred, 90)
    ax[2].scatter(mmpose_coords3d_pred[:, 0], mmpose_coords3d_pred[:, 1], mmpose_coords3d_pred[:, 2], s=6)
    draw_person_limbs_3d_coco(ax[2], mmpose_coords3d_pred, LIMBS_MMPOSE ,color='green', with_face=True)
    for idx, coord in enumerate(mmpose_coords3d_pred):
        ax[2].text(coord[0], coord[1], coord[2], f'{idx}', color='black', fontsize=8)
    ax[2].set_title('MMPose 3D Estimation')
    ax[2].axis('on')

    # # 네 번째 서브플롯에 Kinect SDK 3D 포즈 예측 결과 표시
    # # ax[3] = fig.add_subplot(1, 4, 4, projection='3d')
    # kinect_coords3d_pred = kinect_coords3d_pred - kinect_coords3d_pred[3]
    # kinect_coords3d_pred = rotate_joints_y(kinect_coords3d_pred, 0)
    # kinect_coords3d_pred = rotate_joints_z(kinect_coords3d_pred, 0)
    # kinect_coords3d_pred = rotate_joints_x(kinect_coords3d_pred, 0)
    # for idx, coord in enumerate(kinect_coords3d_pred):
    #     if vis_selected[idx] == 1:
    #         ax[3].scatter(kinect_coords3d_pred[idx, 0], kinect_coords3d_pred[idx, 1], kinect_coords3d_pred[idx, 2], s=10)
    #         ax[3].text(coord[0], coord[1], coord[2], f'{idx}', color='red', fontsize=8)
    # draw_person_limbs_3d_coco(ax[3], kinect_coords3d_pred, LIMBS_KINECT, vis=vis_selected, color='red', with_face=True)
    # ax[3].set_title('Kinect SDK 3D Prediction')
    # ax[3].axis('on')

    # 네 번째 서브플롯에 IHSens 포즈 예측 결과 표시
    kinect_coords3d_pred = kinect_coords3d_pred - kinect_coords3d_pred[3]
    kinect_coords3d_pred = rotate_joints_y(kinect_coords3d_pred, 0)
    kinect_coords3d_pred = rotate_joints_z(kinect_coords3d_pred, 0)
    kinect_coords3d_pred = rotate_joints_x(kinect_coords3d_pred, 0)
    ax[3].scatter(kinect_coords3d_pred[:13, 0], kinect_coords3d_pred[:13, 1], kinect_coords3d_pred[:13, 2], s=5)
    draw_person_limbs_3d_coco(ax[3], kinect_coords3d_pred, LIMBS_KINECT, color='grey')
    for idx, coord in enumerate(kinect_coords3d_pred[:13]):
        ax[3].text(coord[0], coord[1], coord[2], f'{idx}', color='black', fontsize=8)

    # IK 데이터 계산 및 예측된 팔꿈치 위치 시각화
    ik_data = calculate_ik_data(kinect_coords3d_pred)
    left_elbow_predicted = predict_elbow_position(ik_data["left_arm"]["shoulder"], ik_data["left_arm"]["wrist"], ik_data["left_arm"]["arm_length"] / 2)
    right_elbow_predicted = predict_elbow_position(ik_data["right_arm"]["shoulder"], ik_data["right_arm"]["wrist"], ik_data["right_arm"]["arm_length"] / 2)
    ax[3].scatter(left_elbow_predicted[0], left_elbow_predicted[1], left_elbow_predicted[2], s=13, color='red', label='Predicted Left Elbow')
    ax[3].scatter(right_elbow_predicted[0], right_elbow_predicted[1], right_elbow_predicted[2], s=13, color='blue', label='Predicted Right Elbow')

    # IHSens 포즈 예측 결과 표시
    joint_labels = ['Head', 'Left Shoulder', 'Left Wrist', 'Right Shoulder', 'Right Wrist']
    joint_indices = [3, 6, 4, 10, 8] # 머리, 왼손목, 왼어깨, 오른손목, 오른어깨
    joint_colors = ['black', 'green', 'green', 'blue', 'blue']
    for label, idx, color in zip(joint_labels, joint_indices, joint_colors):
        ax[3].scatter(kinect_coords3d_pred[idx, 0], kinect_coords3d_pred[idx, 1], kinect_coords3d_pred[idx, 2], s=8, color=color, label=label)
        ax[3].text(kinect_coords3d_pred[idx, 0], kinect_coords3d_pred[idx, 1], kinect_coords3d_pred[idx, 2], f'{idx}', color='red', fontsize=8)

    ax[3].set_title('3D predicted + IK')
    ax[3].axis('on')
    # ax[4].legend(loc='upper right')

    # 예시: gt_coords3d는 실제 3D 좌표 데이터를 포함해야 합니다.
    ik_data = calculate_ik_data(kinect_coords3d_pred)
    print(ik_data)

    # 결과를 PNG 파일로 저장
    plt.show()
    # plt.tight_layout()
    # plt.savefig(f'{vis_out_dir}/combined_frame_{frame_id}.png')
    # plt.close(fig)


def rotate_joints_z(joints, theta):
        # Convert angle from degrees to radians
        theta_rad = np.radians(theta)
        
        # Create a rotation matrix for the Z-axis
        rotation_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad), 0],
            [np.sin(theta_rad), np.cos(theta_rad), 0],
            [0, 0, 1]
        ])
        
        # Rotate each joint
        rotated_joints = joints @ rotation_matrix
    
        return rotated_joints
def rotate_joints_y(joints, theta):
    # Convert angle from degrees to radians
    theta_rad = np.radians(theta)
    
    # Create a rotation matrix for the Z-axis
    rotation_matrix = np.array([
        [np.cos(theta_rad), 0, np.sin(theta_rad)],
        [0, 1, 0],
        [-np.sin(theta_rad), 0, np.cos(theta_rad)]
    ])
    
    # Rotate each joint
    rotated_joints = joints @ rotation_matrix

    return rotated_joints
def rotate_joints_x(joints, theta):
    # Convert angle from degrees to radians
    theta_rad = np.radians(theta)
    
    # Create a rotation matrix for the Z-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta_rad), -np.sin(theta_rad)],
        [0, np.sin(theta_rad), np.cos(theta_rad)]
    ])
    
    # Rotate each joint
    rotated_joints = joints @ rotation_matrix

    return rotated_joints

def main():
    init_args, call_args, display_alias = parse_args()
    
    if display_alias:
        model_alises = get_model_aliases(init_args['scope'])
        display_model_aliases(model_alises)
    else:
        inferencer = MMPoseInferencer(**init_args)
        pose_results = list(inferencer(**call_args))

        path_to_sdk_json = './tests/data/kinect/pred_sdk.json' 
        path_to_gt_json = './tests/data/kinect/anno.json' 
        image_path = call_args['inputs']

        frame_id = extract_frame_id(image_path)
        if frame_id is None:
            print(f"Could not extract frame number from path: {image_path}")
            return

        frame_data = load_kinect_sdk_predictions(path_to_sdk_json, frame_id)
        anno_data = load_gt(path_to_gt_json, frame_id)
        
        visualize_combined_results(
            image_path=image_path,
            pred_mmpose=pose_results,
            anno_data=anno_data,
            pred_sdk=frame_data,
            frame_id=frame_id,
            vis_out_dir=call_args['vis_out_dir']
        )

        # Check if the frame_id exists in the pred_sdk
        if frame_id in frame_data:
            print(f"Data for frame {frame_id} exists in the pred_sdk")
        else:
            print(f"Data for frame {frame_id} does not exist in the pred_sdk")

if __name__ == '__main__':
    main()