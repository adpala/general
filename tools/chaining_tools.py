import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import copy
import matplotlib.colors as colors
from leap_utils.utils import iswin, ismac, rotate_points
from leap_utils.preprocessing import angles
import os
import deepdish as dd
from videoreader import VideoReader
import flyID_tools as fitools
import xarray_behave as dst
import xarray as xr
import itertools


def getpoints(frame):
    """Get x,y indices of nonzero elements in a matrix."""
    points = np.unravel_index(np.flatnonzero(frame), frame.shape)
    return np.vstack(points).astype(np.float32).T


def segment_cluster(frame, num_clusters=1, term_crit=(cv2.TERM_CRITERIA_EPS, 100, 0.01), init_method=cv2.KMEANS_PP_CENTERS):
    """Cluster points in a frame to find pixel identities."""
    points = getpoints(frame)
    cluster_compactness, labels, centers = cv2.kmeans(points, num_clusters, None, criteria=term_crit, attempts=100, flags=init_method)
    return centers, labels, points


def make_circular_mask(cx, cy, r, n1, n2):
    """Creates a binary mask in a n1 x n2 frame with nonzero values forming a circle of radius r centered in (cx, cy)"""
    y, x = np.ogrid[-cx:n1-cx, -cy:n2-cy]
    mask = x*x + y*y <= r*r
    return mask


def make_ellipse_mask(cx, cy, n1, n2, a, b, orientation=0):
    """Creates a binary mask in a n1 x n2 frame with nonzero values forming an ellipse.

    A focus from the ellipse approximately coincides with the position cx,cy. a and b are the axis
    from the ellipse equation.

    Ellipse is rotated according to the fly's orientation. The ellipse is further displaced, such
    that it approximately falls right in front of the fly in any orientation given.
    The orientation is fixed to follow the angle logic in box_angles, clockwise with a 90 degree offset."""

    ornt = (np.copy(orientation) + 90)*np.pi/180
    y, x = np.ogrid[-cx:n1-cx, -cy:n2-cy]
    ellipse_mask = ((x-np.sqrt(a**2 - b**2)*np.cos(ornt))*np.cos(ornt) + (y-np.sqrt(a**2 - b**2)*np.sin(ornt))*np.sin(ornt))**2/(a*a) + ((x-np.sqrt(a**2 - b**2)*np.cos(ornt))*np.sin(ornt) - (y-np.sqrt(a**2 - b**2)*np.sin(ornt))*np.cos(ornt))**2/(b*b) <= 1
    return ellipse_mask


def get_pixel_ids(bgr_frame, iframe, background, nflies, fly_centers):
    """Finds and assigns the corresponding id to all nonzero pixels in the BG-removed frame provided.
    Creates a new frame with pixel values equal to fly-id +1 (background pixels are 0, therefore the
    pixels are not equal to fly-id, as background pixels would be confused with fly-id = 0)."""

    # kmeans clustering to find flies
    km_centers, km_labels, km_points = segment_cluster(bgr_frame, num_clusters=nflies, term_crit=(cv2.TERM_CRITERIA_EPS, 100, 0.01), init_method=cv2.KMEANS_PP_CENTERS)

    npoints = km_points.shape[0]

    # Find pixel ids
    km_frame = np.zeros_like(bgr_frame)
    km_points = km_points.astype(np.int)
    for ii in range(npoints):
        km_frame[km_points[ii, 0], km_points[ii, 1]] = km_labels[ii]+1

    # reassing pixel ids to match fly id
    new_id = assign_km_ids(nflies, km_centers, fly_centers, iframe)
    new_km_labels = new_id[km_labels] + 1
    new_km_frame = np.zeros_like(bgr_frame)
    km_points = km_points.astype(np.int)
    for ii in range(npoints):
        new_km_frame[km_points[ii, 0], km_points[ii, 1]] = new_km_labels[ii]

    return new_km_frame


def assign_km_ids(nflies, km_centers, fly_centers, iframe):
    """Assigns the id to each cluster of pixels, avoiding repetition, based on distance from the
    cluster center to the fly centers provided (either from tracker or thorax position)."""

    center_distances = np.zeros((nflies, nflies))
    for ifly in range(nflies):
        center_distances[ifly, :] = np.linalg.norm(km_centers[:, :]-fly_centers[iframe, ifly, :], axis=1)

    xx, yy = np.unravel_index(np.argsort(center_distances, axis=None), center_distances.shape)    # xx is track center, yy k_means center
    new_id = np.zeros((nflies))
    taken_yys = []
    taken_xxs = []

    ii = 0
    while len(taken_yys) < nflies:
        if yy[ii] not in taken_yys:
            if xx[ii] not in taken_xxs:
                taken_xxs.append(xx[ii])
                taken_yys.append(yy[ii])
                new_id[yy[ii]] = xx[ii]
        ii += 1

    return np.asarray(new_id, dtype=np.int)


def visualize_km_ids(new_km_frame, iframe, box_angles, heads, tails, circle_offset=40):
    """Plots the new frame with assigned pixel ids. Overlaps a cross for heads (red) and tails (blue).
    Additionally prints the fly-id close to each cluster."""

    nflies = box_angles.shape[1]
    palette = copy(plt.get_cmap('tab20'))
    palette.set_under('white', 1.0)
    levels = np.arange(1, 17)
    norm = colors.BoundaryNorm(levels, ncolors=palette.N)

    plt.figure(figsize=[10, 10])
    plt.imshow(new_km_frame, cmap=palette, norm=norm)
    plt.scatter(heads[iframe, :, 1], heads[iframe, :, 0], c='red', marker='x')
    plt.scatter(tails[iframe, :, 1], tails[iframe, :, 0], c='blue', marker='x')
    for ifly in range(nflies):
        plt.text(heads[iframe][ifly, 1]+30, heads[iframe][ifly, 0]+30, f"{ifly}")


def fly_connectivity(new_km_frame, iframe, heads, box_angles, sense_window=30, plot_matrix=True, print_dict=False, scl=1, bscl=0.5):
    """Finds connectivity of flies based on an elliptical mask in front of the fly. If pixels belonging to another fly are found
    within the ellipse, they are counted for the connectivity.
    Function returns the nflies x nflies matrix with the counts. Self counts are avoided.

    sense_window is the long axis parameter of the ellipse.
    sense_window is multiplied by 2 in the ellipse algorithm, fly body length is approx. 40 pxs in backlight experiments"""

    nflies = box_angles.shape[1]
    overlapping_pixels = np.zeros((nflies, nflies), dtype=np.int)

    for ifly in range(nflies):
        mask = make_ellipse_mask(heads[iframe, ifly, 0], heads[iframe, ifly, 1], n1=new_km_frame.shape[0], n2=new_km_frame.shape[1], a=sense_window, b=sense_window*bscl, orientation=box_angles[iframe, ifly, ...])
        sense_field = np.multiply(new_km_frame, mask)
        unique, counts = np.unique(sense_field[sense_field > 0]-1, return_counts=True)
        counts[unique == ifly] = 0
        overlapping_pixels[ifly, unique.astype(np.int)] = counts
        if print_dict:
            print(ifly, dict(zip(unique, counts)))

    if plot_matrix:
        plt.figure()
        plt.matshow(overlapping_pixels, fignum=0)
        plt.title(f"frame: {iframe}")

    return overlapping_pixels


def ht_distances(heads_pos, tails_pos):
    """Calculates the distance between all heads and tails provided. Can calculate this for multiple frames.
    Function returns a nframe x nflies x nflies matrix.

    An element (k,i,j) from the matrix represents the distance at k-frame between i-head and j-tail."""

    if heads_pos.ndim == 2:
        heads = np.copy(heads_pos)[np.newaxis, ...]
        tails = np.copy(tails_pos)[np.newaxis, ...]
    else:
        heads = np.copy(heads_pos)
        tails = np.copy(tails_pos)

    nflies = heads.shape[1]
    htdist = np.zeros((heads.shape[0], nflies, nflies), dtype=np.float32)
    for ifly in range(nflies):
        for jfly in range(nflies):
            if ifly != jfly:
                htdist[:, ifly, jfly] = np.linalg.norm(heads[:, ifly, :]-tails[:, jfly, :], axis=1)
    return htdist


def orientation_diff(angles):
    """Calculates the difference in orientation between flies. Can calculate this for multiple frames.
    Function returns a nframe x nflies x nflies matrix.
    """

    if angles.ndim == 2:
        box_angles = np.copy(angles)[np.newaxis, ...]
    else:
        box_angles = np.copy(angles)

    box_angles[box_angles < 0] = box_angles[box_angles < 0] + 360

    nflies = box_angles.shape[1]
    nframes = box_angles.shape[0]
    or_diff = np.zeros((nframes, nflies, nflies), dtype=np.float32)
    for ifly in range(nflies):
        for jfly in range(nflies):
            if ifly != jfly:
                or_diff[:, ifly, jfly] = np.abs(box_angles[:, ifly, 0]-box_angles[:, jfly, 0])

    or_diff[or_diff < 0] = or_diff[or_diff < 0] + 360
    or_diff[or_diff > 180] = 360-or_diff[or_diff > 180]

    return or_diff


def get_chain_groups(chain_matrix):
    """Goes through the chain_matrix (the element-wise multiplication of all matrices with parameters
    for chain detection), and checks for common connected members, to find chains (or groups).

    Returns a dictionary with the members of each chain (group)."""

    chain_pairs = getpoints(chain_matrix).astype(np.int)
    groups_dict = {}
    nn = 0
    for ii, jj in chain_pairs:
        found_group = False
        for groupnn in range(nn):
            if ii in groups_dict[f"t{groupnn}"]:
                found_group = True
                if jj not in groups_dict[f"t{groupnn}"]:
                    groups_dict[f"t{groupnn}"].append(jj)
            elif jj in groups_dict[f"t{groupnn}"]:
                found_group = True
                groups_dict[f"t{groupnn}"].append(ii)
        if not found_group:
            groups_dict[f"t{nn}"] = [ii, jj]
            nn += 1

    removed_key = []
    for a, b in itertools.combinations(groups_dict.keys(), 2):
        if not set(groups_dict[a]).isdisjoint(groups_dict[b]):
            groups_dict[a] = list(set().union(groups_dict[a], groups_dict[b]))
            groups_dict[b] = []
            removed_key.append(b)

    for rk in removed_key:
        del groups_dict[rk]

    nn = 0
    for k in sorted(groups_dict, key=lambda k: len(groups_dict[k]), reverse=True):
        groups_dict[nn] = groups_dict.pop(k)
        nn += 1

    return groups_dict


def load_all_forchain(expID, expsetup: str = 'chainingmic'):
    """ expsetup can be 'chainingmic', 'backlight' or 'backlight_touch'. """

    # Adjust paths to computer
    if iswin():
        root = 'Z:/#Common/'
    elif ismac():
        root = '/Volumes/ukme04/#Common/'
    else:
        root = '/scratch/clemens10/'
    bl_dat_folder = f"{root}{expsetup}/dat"
    bl_res_folder = f"{root}{expsetup}/res"

    # File paths
    video_path = f"{bl_dat_folder}/{expID}/{expID}.mp4"
    if not os.path.exists(video_path):
        video_path = f"{bl_dat_folder}.processed/{expID}/{expID}.mp4"
    vr = VideoReader(video_path)

    print(f"video frames: {vr.number_of_frames}")

    # loading tracks

    trackfixed_path = f"{bl_res_folder}/{expID}/{expID}_tracks_fixed.h5"
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

    # background (correcting to make it an image with 3 channels)
    background = fixtrack_data['background'][..., np.newaxis]
    background = np.concatenate((background, background, background), axis=2)

    return vr, centers, box_centers, nflies, heads, tails, box_angles, background


def old_chain_analysis(vr, iframes, background, centers, heads, tails, box_angles, sense_window=60, bscl=0.75, ordiff_thr=90, htdist_thr=100, conn_thr=0, do_conn=True, do_dist=True, do_or=False, do_ang=True, thr=0.3, save_frames=False, ang_thr=70):
    frames_list = list(vr[iframes.tolist()])
    bgr_frames = fitools.create_bgr(frames_list, background, thr=thr)
    fly_centers = centers[:, 0, :, :]
    nflies = fly_centers.shape[1]
    nframes = iframes.shape[0]

    all_chain_matrix = np.zeros((nframes, nflies, nflies))
    if do_conn:
        all_fly_conn = np.zeros((nframes, nflies, nflies))
    if do_dist:
        all_htdist = np.zeros((nframes, nflies, nflies))
    if do_or:
        all_or_diff = np.zeros((nframes, nflies, nflies))
    if do_ang:
        all_ang_mat = np.zeros((nframes, nflies, nflies))
    if save_frames:
        all_new_frames = np.zeros_like(bgr_frames)

    results_dict = {}

    for jj, bgr_frame in enumerate(bgr_frames):

        if nframes > 200:
            if jj % (nframes/10) == 0:
                print(f'processing... {100*jj/nframes} %')

        # segmented frame
        new_km_frame = get_pixel_ids(bgr_frame, iframes[jj], background, nflies, fly_centers)

        # calculate and merge conditions for detection
        if do_conn or do_dist or do_or:
            chain_matrix = np.ones((nflies, nflies))
            if do_conn:
                # fly connectivity
                fly_conn = fly_connectivity(new_km_frame, iframes[jj], heads, box_angles, sense_window, bscl=bscl, plot_matrix=False)
                chain_matrix = np.multiply(chain_matrix, fly_conn > conn_thr)
                all_fly_conn[jj, ...] = fly_conn
            if do_dist:
                # head-tail distance
                htdist = ht_distances(heads[iframes[jj], ...], tails[iframes[jj], ...])
                chain_matrix = np.multiply(chain_matrix, htdist[0, ...] < htdist_thr)
                all_htdist[jj, ...] = htdist[0, ...]
            if do_or:
                # orientation difference
                or_diff = orientation_diff(box_angles[iframes[jj], ...])
                chain_matrix = np.multiply(chain_matrix, or_diff[0, ...] < ordiff_thr)
                all_or_diff[jj, ...] = or_diff[0, ...]
            if do_ang:
                ang_mat = angle_between_flies(heads[iframes[jj], ...], tails[iframes[jj], ...])
                chain_matrix = np.multiply(chain_matrix, ang_mat[0, ...] < ang_thr)
                all_ang_mat[jj, ...] = ang_mat[0, ...]
            # print(f"{get_chain_groups(chain_matrix)}")
        else:
            print('all conditions are false, choose a condition (do_X = True) to detect chains.')
            chain_matrix = np.zeros((nflies, nflies))

        all_chain_matrix[jj, ...] = chain_matrix
        if save_frames:
            all_new_frames[jj, ...] = new_km_frame

    if do_conn or do_dist or do_or:
        results_dict['chain_matrix'] = all_chain_matrix
        if do_conn:
            results_dict['fly_conn'] = all_fly_conn
        if do_dist:
            results_dict['htdist'] = all_htdist
        if do_or:
            results_dict['or_diff'] = all_or_diff
        if do_ang:
            results_dict['ang_mat'] = all_ang_mat
    if save_frames:
        results_dict['new_frames'] = all_new_frames

    return results_dict


def angle_between(x1, y1, x2, y2):
    dot = x1*x2 + y1*y2
    det = x1*y2 - y1*x2
    angle = np.arctan2(det, dot)
    return angle*180/np.pi


def angle_between_flies(heads_pos, tails_pos):
    """Calculates the between flies (Head-Tail vector vs Own axis). Can calculate this for multiple frames.
    Function returns a nframe x nflies x nflies matrix.

    An element (k,i,j) from the matrix represents the angle at k-frame between i-head and j-tail, relative to i-axis."""

    if heads_pos.ndim == 2:
        heads = np.copy(heads_pos)[np.newaxis, ...]
        tails = np.copy(tails_pos)[np.newaxis, ...]
    else:
        heads = np.copy(heads_pos)
        tails = np.copy(tails_pos)

    nflies = heads.shape[1]
    angles_matrix = np.zeros((heads.shape[0], nflies, nflies), dtype=np.float32)
    for ifly in range(nflies):
        for jfly in range(nflies):
            if ifly != jfly:
                angles_matrix[:, ifly, jfly] = angle_between(heads[:, ifly, 1]-tails[:, ifly, 1], heads[:, ifly, 0]-tails[:, ifly, 0], tails[:, jfly, 1]-heads[:, ifly, 1], tails[:, jfly, 0]-heads[:, ifly, 0])
    return angles_matrix


def chain_description(data, box_centers):

    # Chaining properties

    chain_matrices = data['chain_matrix']
    nflies = chain_matrices.shape[1]
    groups_per_frame = np.zeros((chain_matrices.shape[0]))
    chain_score_per_frame = np.zeros((chain_matrices.shape[0]))
    chflies_per_frame = np.zeros((chain_matrices.shape[0]))
    for ii, chain_matrix in enumerate(chain_matrices):
        chgroups = get_chain_groups(chain_matrix)
        groups_per_frame[ii] = len(chgroups)
        if groups_per_frame[ii] != 0:
            chflies_per_frame[ii] = np.sum(np.asarray([len(chgroups[gg]) for gg in chgroups]))
            chain_score_per_frame[ii] = np.mean(np.asarray([len(chgroups[gg]) for gg in chgroups]))/nflies

    # RMSD

    iframes = data['iframes']
    rmsd = np.zeros(iframes[-1]-iframes[0])
    for ii, ff in enumerate(range(iframes[0], iframes[-1])):
        for ifly in range(nflies):
            rmsd[ii] += np.sum(np.square(np.linalg.norm(box_centers[ff, :, :]-box_centers[ff, ifly, :], axis=1)))
    rmsd = np.sqrt(rmsd/(nflies*(nflies-1)))

    return groups_per_frame, chain_score_per_frame, chflies_per_frame, rmsd


def plot_chain_description(iframes, groups_per_frame, chflies_per_frame, chain_score_per_frame, rmsd, save_fig=False):
    plt.rcParams['font.size'] = 16

    plt.figure(figsize=[15, 12])
    plt.subplot(411)
    plt.plot(iframes, groups_per_frame)
    plt.ylabel('# Groups')
    plt.gca().set_xticklabels([])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplot(412)
    plt.plot(iframes, chflies_per_frame)
    plt.ylabel('# Flies')
    plt.gca().set_xticklabels([])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().set_xticklabels([])
    plt.subplot(413)
    plt.plot(iframes, chain_score_per_frame)
    plt.ylabel('Chain score')
    plt.gca().set_xticklabels([])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplot(414)
    plt.plot(range(iframes[0], iframes[-1]), rmsd)
    plt.ylabel('RMSD')
    plt.xlabel('time (frames)')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
#     plt.savefig(f'chainscore_perframe_{expID}.png', bbox_inches='tight', transparent=True)
#     plt.show()


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)


def find_longest(triads_data):

    longest_triad = np.array([0])
    longest_triad_key = None
    for key in triads_data.keys():
        consecutive_triads = consecutive(triads_data[key])
        max_triad = max(consecutive_triads, key=lambda x: x.size)
        if max_triad.size > longest_triad.size:
            longest_triad = max_triad
            longest_triad_key = key

    return longest_triad_key, longest_triad


def common_field(fly_centers, box_angles, iframes):
    """ Calculate the field that each fly perceives (relative position distribution of other flies around target fly) """

    nflies = fly_centers.shape[1]
    nframes = iframes.shape[0]

    # get relative positions
    no_rot_rel_centers = np.zeros((nframes, nflies, nflies, 2))
    rel_centers = np.zeros((nframes, nflies, nflies, 2))
    for ifly in range(nflies):
        for ii, iframe in enumerate(iframes):
            no_rot_rel_centers[ii, ifly, :, :] = fly_centers[iframe, :, :]-fly_centers[iframe, ifly, :]
            rel_centers[ii, ifly, :, :] = np.asarray(rotate_points(no_rot_rel_centers[ii, ifly, :, 1], no_rot_rel_centers[ii, ifly, :, 0], box_angles[iframe, ifly, 0])).T

    all_rel_center = np.empty((0, 2))
    for ifly in range(nflies):
        rel_centers[:, ifly, ifly, :] = np.nan
        rel_center_ifly = np.copy(rel_centers[:, ifly, :, :]).reshape((rel_centers.shape[0]*rel_centers.shape[1], 2))
        rel_center_ifly = rel_center_ifly[~np.isnan(rel_center_ifly).any(axis=1)]
        small_rel_center_ifly = rel_center_ifly[np.linalg.norm(rel_center_ifly, axis=1) > 0, :]
        all_rel_center = np.concatenate((all_rel_center, small_rel_center_ifly[:, ::-1]), axis=0)

    return all_rel_center


def single_common_field(target_fly, fly_centers, box_angles, iframes):
    """ Calculate the field that each fly perceives (relative position distribution of other flies around target fly) """

    nflies = fly_centers.shape[1]
    nframes = iframes.shape[0]

    # get relative positions
    no_rot_rel_centers = np.zeros((nframes, nflies, nflies, 2))
    rel_centers = np.zeros((nframes, nflies, nflies, 2))
    ifly = target_fly
    for ii, iframe in enumerate(iframes):
        no_rot_rel_centers[ii, ifly, :, :] = fly_centers[iframe, :, :]-fly_centers[iframe, ifly, :]
        rel_centers[ii, ifly, :, :] = np.asarray(rotate_points(no_rot_rel_centers[ii, ifly, :, 1], no_rot_rel_centers[ii, ifly, :, 0], box_angles[iframe, ifly, 0])).T

    all_rel_center = np.empty((0, 2))
    ifly = target_fly

    rel_centers[:, ifly, ifly, :] = np.nan
    rel_center_ifly = np.copy(rel_centers[:, ifly, :, :]).reshape((rel_centers.shape[0]*rel_centers.shape[1], 2))
    rel_center_ifly = rel_center_ifly[~np.isnan(rel_center_ifly).any(axis=1)]
    small_rel_center_ifly = rel_center_ifly[np.linalg.norm(rel_center_ifly, axis=1) > 0, :]
    all_rel_center = np.concatenate((all_rel_center, small_rel_center_ifly[:, ::-1]), axis=0)

    return all_rel_center


def load_metrics(datename, expsetup='backlight', data_folder="Z:/apalaci/code/dat", overwrite=False, lazy=True, save_file=False, target_sampling_rate=1000):
    # Load experiment
    root = f"Z:/#Common/{expsetup}"
    data_path = f"{data_folder}/{datename}/{datename}.zarr"
    metrics_data_path = f"{data_folder}/{datename}/{datename}_metrics.zarr"

    # Prepare folder for data in case it doesn't exist
    exp_dat_folder = f"{data_folder}/{datename}"
    if not os.path.isdir(exp_dat_folder):
        os.mkdir(exp_dat_folder)
        print(f"   created directory: {exp_dat_folder}")

    # Load or create data xarray
    if (not os.path.exists(data_path)) or overwrite:
        print(f'   assembling data from {datename}')
        dataset = dst.assemble(datename, root=root, target_sampling_rate=target_sampling_rate)
        if save_file:
            print(f'   saving to {data_path}')
            dst.save(data_path, dataset)
        else:
            print(f'   saving option off')

    else:
        print(f'   loading data from {data_path}')
        dataset = dst.load(data_path, lazy=lazy)

    # Load or create metrics dataset
    if (not os.path.exists(metrics_data_path)) or overwrite:
        # Assemble metrics xarray
        print(f'   assembling metrics data from {datename}')
        metrics_dataset = dst.assemble_metrics(dataset)
        if save_file:
            print(f'   saving to {metrics_data_path}')
            dst.save(metrics_data_path, metrics_dataset)
        else:
            print(f'   saving option off')
    else:
        # Load metrics xarray from file
        print(f'   loading metrics data from {metrics_data_path}')
        metrics_dataset = dst.load(metrics_data_path, lazy=lazy)

    return metrics_dataset, dataset


def chain_analysis(metrics_dataset, dist_thr=100, ang_thr=70, or_thr=100):

    # Load data
    print(f"   loading specific metrics...")
    rel_orientations = metrics_dataset.rel_features.sel(relative_features='relative_orientation')
    dist = metrics_dataset.rel_features.sel(relative_features='distance')
    rel_angle = metrics_dataset.rel_features.sel(relative_features='relative_angle')
    # vel_mag = metrics_dataset.abs_features.sel(absolute_features='velocity_magnitude')

    # Correct rel_orientation to be -180 to 180
    rel_orientations = rel_orientations - ((rel_orientations+180)//360)*360

    # Deduce variables from data dimensions
    nflies = dist.shape[1]
    nframes = dist.shape[0]

    # Create chain matrices based on conditions
    print(f"   calculating chain matrices...")
    chain_matrices = np.ones((nframes, nflies, nflies))
    chain_matrices = np.multiply(chain_matrices, dist < dist_thr)
    chain_matrices = np.multiply(chain_matrices, np.abs(rel_angle) < ang_thr)
    chain_matrices = np.multiply(chain_matrices, np.abs(rel_orientations) < or_thr)

    chain_matrices = chain_matrices.astype(bool)

    chain_dataset = xr.DataArray(data=chain_matrices, dims=['time', 'flies', 'relative_flies'], coords={'time': metrics_dataset.time})

    print(f"   done")

    return chain_matrices, chain_dataset


def get_my_lengths(chain_matrices):

    my_chain_matrices = chain_matrices[np.sum(chain_matrices, axis=(1, 2)) > 0, ...]

    all_lengths = []
    for ii, chain_matrix in enumerate(my_chain_matrices):
        chgroups = get_chain_groups(chain_matrix)
        all_lengths.extend([len(chgroups[gg]) for gg in chgroups])

    all_lengths = np.asarray(all_lengths)

    return all_lengths


def plot_lengths(lengths):

    plt.rcParams['font.size'] = 16
    plt.hist(lengths, rwidth=0.8, align='left', bins=np.arange(2, lengths.max()+2))
    plt.yscale('log')
    plt.ylabel('log(Count)')
    plt.xlabel('length (# of flies in a chain)')
    plt.title('Chain length distribution')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks(np.arange(2, lengths.max()+1))
    plt.show()


def triad_analysis(dataset, chain_matrices, time_step=1000, verbose=True, target_sampling_rate=1000):

    idx_step = int(time_step*target_sampling_rate/1000)
    times = dataset.nearest_frame.time[::idx_step].values
    # frame_numbers = np.unique(dataset.nearest_frame.sel(time=times).data)
    # vtimes = [dataset.time[dataset.nearest_frame == frame_number][0].values for frame_number in frame_numbers]

    triads_data = {}
    # my_chain_matrices = chain_matrices.loc[vtimes, ...].values
    my_chain_matrices = chain_matrices.loc[times, ...].values

    for ii, chain_matrix in enumerate(my_chain_matrices):
        chgroups = get_chain_groups(chain_matrix)
        group_lengths = np.asarray([len(chgroups[gg]) for gg in chgroups])
        for ngg in np.where(group_lengths > 2)[0]:
            for subset in itertools.permutations(chgroups[ngg], 3):
                if chain_matrix[subset[0], subset[1]]*chain_matrix[subset[1], subset[2]] == 1:
                    if f"{subset[0]:02d}{subset[1]:02d}{subset[2]:02d}" in triads_data:
                        triads_data[f"{subset[0]:02d}{subset[1]:02d}{subset[2]:02d}"].append(ii)
                    else:
                        triads_data[f"{subset[0]:02d}{subset[1]:02d}{subset[2]:02d}"] = [ii]

    longest_triad_key, longest_triad = find_longest(triads_data)

    if verbose:
        max_key = max(triads_data, key=lambda x: len(set(triads_data[x])))
        print(f"Most frequent Triad: {max_key}, {len(triads_data[max_key])} sample points (every {time_step} ms)")
        print(f"Longest triad: {longest_triad_key}, duration: {time_step*longest_triad.size} ms, at {time_step*longest_triad[0]}-{time_step*longest_triad[-1]} ms")

    return triads_data, longest_triad_key, longest_triad


def segment_continuous_triads(triads_data, stepsize=1):

    segmented_triads = {}
    for key in triads_data.keys():
        consecutive_triads = np.split(triads_data[key], np.where(np.diff(triads_data[key]) > stepsize)[0] + 1)
        for nt, ct in enumerate(consecutive_triads):
            segmented_triads[f"{key}_{nt:02d}"] = list(ct)

    return segmented_triads


def get_flynumber(key):
    back_fly = int(key[0:2])
    middle_fly = int(key[2:4])
    front_fly = int(key[4:6])
    return back_fly, middle_fly, front_fly
