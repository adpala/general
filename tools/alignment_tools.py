import numpy as np
import deepdish as dd
from videoreader import VideoReader
from leap_utils.utils import iswin, ismac, unflatten, flatten, rotate_points
from leap_utils.preprocessing import export_boxes, angles
from leap_utils.plot import boxpos
import matplotlib.pyplot as plt
import os


def load_all_with_poses(expID, expsetup: str = 'chainingmic'):
    """ expsetup can be 'chainingmic', 'backlight' or 'backlight_touch'. """

    # Adjust paths to computer
    if iswin():
        root = 'Z:/#Common/'
    elif ismac():
        root = '/Volumes/ukme04/#Common/'
    else:
        root = '/scratch/clemens10/'
    dat_folder = f"{root}{expsetup}/dat"
    res_folder = f"{root}{expsetup}/res"

    # File paths
    video_path = f"{dat_folder}/{expID}/{expID}.mp4"
    if not os.path.exists(video_path):
        video_path = f"{dat_folder}.processed/{expID}/{expID}.mp4"
    vr = VideoReader(video_path)

    print(f"video frames: {vr.number_of_frames}")

    # loading tracks

    trackfixed_path = f"{res_folder}/{expID}/{expID}_tracks_fixed.h5"
    fixtrack_data = dd.io.load(trackfixed_path)

    # Get positions, chamber corrections, and number of flies
    centers = fixtrack_data['centers']
    chbb = fixtrack_data['chambers_bounding_box'][:]
    box_centers = centers[:, 0, :, :]
    box_centers = box_centers + chbb[1][0][:]
    nflies = box_centers.shape[1]
    print(f"nflies: {nflies}")
    print(f"number of center frames: {box_centers.shape[0]}")

    # Get head-tail and angles
    tracks = fixtrack_data['lines']
    tails = tracks[:, 0, :, 0, ::-1]   # nframe, fly id, coordinates
    heads = tracks[:, 0, :, 1, ::-1]   # nframe, fly id, coordinates
    heads = heads + chbb[1][0][:]   # nframe, fly id, coordinates
    tails = tails + chbb[1][0][:]   # nframe, fly id, coordinates
    box_angles = angles(tails, heads)
    print(box_angles.shape)

    # background (correcting to make it an image with 3 channels)
    background = fixtrack_data['background'][..., np.newaxis]
    background = np.concatenate((background, background, background), axis=2)

    # poses
    try:
        poses_path = f"{res_folder}/{expID}/{expID}_poses.h5"
        poses_data = dd.io.load(poses_path)
        poses = poses_data['positions'].astype(np.float64)
    except FileNotFoundError:
        print('no poses found')
        poses = None

    return vr, centers, box_centers, nflies, heads, tails, box_angles, background, poses


def fix_translation(box_centers, box_size_length, box_angles, poses):
    nframes = box_centers.shape[0]
    nflies = box_centers.shape[1]
    nbodyparts = poses.shape[1]
    new_box_centers = np.copy(box_centers)
    new_poses = np.copy(poses).astype(np.float64)
    thorax_center_diff = unflatten(box_size_length/2-poses[:, 8, :], nflies)
    for ii in range(nbodyparts):
        new_poses[:, ii, :] = new_poses[:, ii, :] + flatten(thorax_center_diff)
    new_diff = np.zeros_like(thorax_center_diff)
    for ii in range(nframes):
        for ifly in range(nflies):
            new_diff[ii, ifly, :] = rotate_points(thorax_center_diff[ii, ifly, 0], thorax_center_diff[ii, ifly, 1], box_angles[ii, ifly, :], origin=(0, 0))

    new_box_centers = new_box_centers - new_diff

    return new_box_centers, new_poses


def fix_rotation(box_size_length, box_angles, poses):
    nframes = box_angles.shape[0]
    nflies = box_angles.shape[1]
    nbodyparts = poses.shape[1]
    new_box_angles = np.copy(box_angles)
    new_poses = np.copy(poses).astype(np.float64)
    calc_angles_head_thorax = unflatten(90 + np.arctan2(poses[:, 0, 0] - poses[:, 8, 0], poses[:, 0, 1] - poses[:, 8, 1]) * 180 / np.pi, nflies)[..., np.newaxis]
    new_box_angles = calc_angles_head_thorax+new_box_angles

    for ii in range(nframes):
        for jj in range(nbodyparts):
            new_poses[ii, jj, :] = rotate_points(new_poses[ii, jj, 0], new_poses[ii, jj, 1], -flatten(calc_angles_head_thorax)[ii], origin=(box_size_length/2, box_size_length/2))

    return new_box_angles, new_poses


def inspect_all_flies(vr, iframes, nflies, box_centers, box_angles, background, poses=None):

    # extract images from video
    frame_list = list(vr[iframes.tolist()])

    # export boxes (it will alternate boxes from all flies in the video)
    boxes, _, _ = export_boxes(frames=frame_list, box_centers=box_centers, box_size=[120, 120], box_angles=box_angles, background=background)

    plt.figure(figsize=[4*nflies, 4*iframes.shape[0]])
    for jj in range(iframes.shape[0]):
        for ifly in range(nflies):
            plt.subplot(iframes.shape[0], nflies, jj*nflies+ifly+1)
            if np.all(poses) is None:
                plt.imshow(boxes[ifly::nflies, :, :, :3][jj], cmap=plt.get_cmap('Greys'))
            else:
                boxpos(boxes[ifly::nflies, :, :, :3][jj], poses[ifly::nflies, :, :][jj])
            plt.scatter(60, 60, color='r', marker='x')
            plt.plot([60, 60], [10, 110], c='blue')
            # plt.axis('off')
        plt.subplot(iframes.shape[0], nflies, jj*nflies+1)
        plt.title(f"frame {iframes[jj]}")
