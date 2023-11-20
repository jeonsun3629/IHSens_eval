# Copyright (c) OpenMMLab. All rights reserved.
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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs',
        type=str,
        nargs='?',
        default = './tests/data/kinect/color/1_00000000.png',
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


LIMBS_MMPOSE = np.array([[8, 11], [11, 12], [12, 13],  # right arm
                        [0, 4], [4, 5], [5, 6],  # right leg
                        [8, 14], [14, 15], [15, 16],  # left arm
                        [0, 1], [1, 2], [2, 3],  # left leg
                        [0, 7], [7, 8], [8, 9], [9, 10]])  # head

LIMBS_KINECT = np.array([[20, 8], [8, 9], [9, 10], # right arm
                         [0, 16], [16, 17], [17, 18], # right leg
                         [0, 12,], [12, 13], [13, 14], # left leg
                         [20, 4], [4, 5], [5, 6], # left arm
                         [0, 1], [1, 0], [1, 2], [2, 3]]) # head


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

def visualize_combined_results(image_path, pred_mmpose, pred_sdk, frame_id, vis_out_dir):
    # 이미지 로드
    image = plt.imread(image_path)
   
    # mmpose 3D 포즈 추정 결과 추출
    mmpose_coords3d_pred = np.array(pred_mmpose[frame_id]['predictions'][0][0]['keypoints'])
    print(f"Coords3D MMPose Pred for frame {frame_id}: {mmpose_coords3d_pred}")

    # Kinect SDK 3D 포즈 예측 결과 추출
    kinect_coords3d_pred = [np.array(coords) for coords in pred_sdk[frame_id][0][0] if len(coords) == 3]
    if not kinect_coords3d_pred:
        print(f"No 3D coordinates to visualize for frame {frame_id}")
        return

    # 1차원 리스트의 리스트를 2차원 NumPy 배열로 변환합니다.
    kinect_coords3d_pred = np.array(kinect_coords3d_pred)

    # 스케일 비율을 사용하여 Kinect SDK 좌표 조정
    kinect_coords3d_pred_scaled = scale_coords_to_reference(mmpose_coords3d_pred, kinect_coords3d_pred, joint_mapping)

    # Figure 및 Axes 설정
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # 3개의 서브플롯 생성

    # 첫 번째 서브플롯에 2D 이미지 표시
    ax[0].imshow(image)
    ax[0].set_title('2D Image')
    ax[0].axis('off')  # 축 숨기기

    # 두 번째 서브플롯에 MMPose 3D 포즈 추정 결과 표시
    ax[1] = fig.add_subplot(1, 3, 2, projection='3d')
    ax[1].scatter(mmpose_coords3d_pred[:, 0], mmpose_coords3d_pred[:, 1], mmpose_coords3d_pred[:, 2], s=60)
    draw_limbs_3d(ax[1], mmpose_coords3d_pred, LIMBS_MMPOSE, color='blue')
    for idx, coord in enumerate(mmpose_coords3d_pred):
        ax[1].text(coord[0], coord[1], coord[2], f'{idx}', color='blue')
    ax[1].set_title('MMPose 3D Estimation')
    ax[1].axis('on')

    # 세 번째 서브플롯에 Kinect SDK 3D 포즈 예측 결과 표시
    ax[2] = fig.add_subplot(1, 3, 3, projection='3d')
    ax[2].scatter(kinect_coords3d_pred_scaled[:, 0], kinect_coords3d_pred_scaled[:, 1], kinect_coords3d_pred_scaled[:, 2], s=60)
    draw_limbs_3d(ax[2], kinect_coords3d_pred_scaled, LIMBS_KINECT, color='red') 
    for idx, coord in enumerate(kinect_coords3d_pred_scaled):
        ax[2].text(coord[0], coord[1], coord[2], f'{idx}', color='red')
    ax[2].set_title('Kinect SDK 3D Prediction')
    ax[2].axis('on')
    
    # 결과를 PNG 파일로 저장
    plt.tight_layout()
    plt.savefig(f'{vis_out_dir}/combined_frame_{frame_id}.png')
    plt.close(fig)

def main():
    init_args, call_args, display_alias = parse_args()
    
    if display_alias:
        model_alises = get_model_aliases(init_args['scope'])
        display_model_aliases(model_alises)
    else:
        inferencer = MMPoseInferencer(**init_args)
        pose_results = list(inferencer(**call_args))

        path_to_sdk_json = './tests/data/kinect/pred_sdk.json' 
        image_path = call_args['inputs']

        frame_id = extract_frame_id(image_path)
        if frame_id is None:
            print(f"Could not extract frame number from path: {image_path}")
            return

        frame_data = load_kinect_sdk_predictions(path_to_sdk_json, frame_id)
        
        visualize_combined_results(
            image_path=image_path,
            pred_mmpose=pose_results,
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