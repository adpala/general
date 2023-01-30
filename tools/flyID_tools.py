import numpy as np
import matplotlib.pyplot as plt
from videoreader import VideoReader
import cv2
import scipy.ndimage as sci
# import skimage.segmentation
import deepdish as dd
from leap_utils.preprocessing import export_boxes, angles
import os
import socket


def find_epoch_end(case_indx, trace, jump_thr, nwindow=5, e_extra_end=0):

    # looks for the end of the epoch of mistakes, and chooses a frame which passes the next condition:
    #        the end of an epoch should be followed by a frame which is consequently followed by at least nwindow frames without jumps, according to the jump_thr threshold.
    # Index logic is based on jump_check logic, not on real frames logic (i.e. index = real_frame_index - frame_range[0])

    not_yet = True
    ii = 0
    end_indx = 0
    while not_yet and end_indx + nwindow + 1 < trace.shape[0]:
        end_indx = case_indx+ii+1
        if all(trace[end_indx:end_indx+nwindow+1] < jump_thr):
            not_yet = False
        else:
            ii += 1

    end_indx += e_extra_end

    return end_indx


def find_good_ref(case_indx, trace, jump_thr, frame_lim=50, nwindow=5):

    # looks for a good position of reference in previous frames, and chooses one which passes the next condition:
    #        a good reference must not contain jumps in itself and its previous nwindow frames, according to the jump_thr threshold.
    # Index logic is based on jump_check logic, not on real frames logic (i.e. index = real_frame_index - frame_range[0])

    not_yet = True
    ii = 0
    while not_yet and ii < frame_lim:
        good_indx = case_indx-ii-1
        if all(trace[good_indx-nwindow+1:good_indx+1] < jump_thr):
            not_yet = False
        else:
            ii += 1

    return good_indx


def check_jumps(nflies, frame_range, box_centers, jump_thr):
    jump_check = np.zeros((nflies, frame_range.shape[0]))
    for ifly in range(nflies):
        jump_check[ifly, 1:] = np.linalg.norm(box_centers[frame_range, ...][1:, ifly, :]-box_centers[frame_range, ...][:-1, ifly, :], axis=1)

    print(f"min/avg/max jump: {np.amin(jump_check)}/{np.mean(jump_check)}/{np.amax(jump_check)}")

    # get indexes above threshold

    jump_check_indxs = np.asarray(np.where(jump_check > jump_thr)).T
    jump_check_indxs = np.concatenate((jump_check_indxs, jump_check_indxs[:, [1]]+frame_range[0]), axis=1)  # adding dimension with corrected indx for referencing real frame number
    print(f"Threshold {jump_thr} pixels, found {jump_check_indxs.shape[0]} cases")
    jump_check_indxs = jump_check_indxs[jump_check_indxs[:, 1].argsort()]    # sort cases from earliest to latest

    return jump_check, jump_check_indxs


def find_closest_trace(traces, ref_frame, end_frame, ifly):
    # traces are [nframes x nflies x coordinates]
    # returns the frame and the fly id of the best match

    jj = 0
    toofar = True
    while toofar and jj < 50:
        fly_dists = np.linalg.norm(traces[end_frame+jj, :, :]-traces[ref_frame, ifly, :], axis=1)
        if fly_dists.min() < 10:
            toofar = False
        else:
            jj += 1

    return end_frame+jj, np.argmin(fly_dists)


def inspect_all_flies(vr, iframes, nflies, box_centers, box_angles, background):

    # extract images from video
    frame_list = list(vr[iframes.tolist()])

    # export boxes (it will alternate boxes from all flies in the video)
    boxes, _, _ = export_boxes(frames=frame_list, box_centers=box_centers[iframes, ...], box_size=[80, 80], box_angles=box_angles[iframes, ...], background=background)

    plt.figure(figsize=[4*nflies, 4*iframes.shape[0]])
    for jj in range(iframes.shape[0]):
        for ifly in range(nflies):
            plt.subplot(iframes.shape[0], nflies, jj*nflies+ifly+1)
            plt.imshow(boxes[ifly::nflies, :, :, :3][jj], cmap=plt.get_cmap('Greys'))
            plt.axis('off')
        plt.subplot(iframes.shape[0], nflies, jj*nflies+1)
        plt.title(f"frame {iframes[jj]}")


def match_ids(nflies, all_positions, ref_frame, end_frame, error_detected):

    fly_dists = np.zeros((nflies, nflies))
    for ifly in range(nflies):
        fly_dists[ifly, :] = np.linalg.norm(all_positions[end_frame, :, :]-all_positions[ref_frame, ifly, :], axis=1)

    xx, yy = np.unravel_index(np.argsort(fly_dists, axis=None), fly_dists.shape)    # xx is previous, yy i
    new_id = np.zeros((nflies))
    taken_yys = []
    taken_xxs = []

    already_good = np.where(np.sum(error_detected[:, ref_frame:end_frame], axis=1) == 0)[0]
    print(f"already good: {already_good}")

    for ii in already_good:
        taken_xxs.append(ii)
        taken_yys.append(ii)
        new_id[ii] = ii

    ii = 0

    while len(taken_yys) < 7:
        if yy[ii] not in taken_yys:
            if xx[ii] not in taken_xxs:
                taken_xxs.append(xx[ii])
                taken_yys.append(yy[ii])
                new_id[xx[ii]] = yy[ii]
        ii += 1

    return new_id.astype(int)


def evaluate_jumps(nflies, frame_range, all_positions, jump_thr, error_pad=20):
    # jump check
    jump_check, jump_check_indxs = check_jumps(nflies, frame_range, all_positions, jump_thr)

    # detect errors (jumps)
    error_detected = jump_check > jump_thr
    # cumulative error (whether an error happened to any fly at a specific frame)
    cum_error = np.sum(jump_check > jump_thr, axis=0)

    if error_pad > 0:
        # pad error frames, to cover for undetected errors before or after detection
        exploded_cum_error = np.zeros_like(cum_error)
        for ii in np.where(cum_error)[0]:
            if ii+error_pad <= exploded_cum_error.shape[0]:
                exploded_cum_error[ii-error_pad:ii+error_pad] = 1
            else:
                exploded_cum_error[ii-error_pad:] = 1
    else:
        exploded_cum_error = cum_error

    # find onsets and offsets of padded errors
    error_onoffset = np.zeros_like(exploded_cum_error)
    error_onoffset[1:] = abs(exploded_cum_error[1:]-exploded_cum_error[:-1]) > 0
    error_edges_frames = np.where(error_onoffset)[0]

    print(f"number of error edges: {error_edges_frames.shape[0]}")

    return error_detected, error_edges_frames


def switch_traces(positions, angles, new_id, ref_frame):
    new_positions = np.copy(positions)
    new_angles = np.copy(angles)

    for ii, jj in enumerate(new_id):
        if ii != jj:
            new_positions[ref_frame:, ii, :] = positions[ref_frame:, jj, :]
            new_angles[ref_frame:, ii, :] = angles[ref_frame:, jj, :]

    return new_positions, new_angles


def replace_with_ref(positions, angles, new_id, ref_frames, end_frames):

    for ii in range(ref_frames.shape[0]):
        ref_frame = ref_frames[ii]
        end_frame = end_frames[ii]
        positions[ref_frame:end_frame, :, :] = positions[ref_frame, :, :]
        angles[ref_frame:end_frame, :, :] = angles[ref_frame, :, :]

    return positions, angles


def inspect_problem_frame(vr, selected_frame, ifly, all_positions, box_angles, background, nflies):
    iboxframes = np.arange(selected_frame-5, selected_frame+6)
    frame_list = list(vr[iboxframes.tolist()])
    boxes, _, _ = export_boxes(frames=frame_list, box_centers=all_positions[iboxframes, ...], box_size=[80, 80], box_angles=box_angles[iboxframes, ...], background=background)

    plt.figure(figsize=[4*iboxframes.shape[0], 4])
    for jj in range(iboxframes.shape[0]):
        plt.subplot(1, iboxframes.shape[0], jj+1)
        plt.imshow(boxes[ifly::nflies, :, :, :3][jj], cmap=plt.get_cmap('Greys'))
        plt.axis('off')


def my_erode(frame, kernel_size=3):
    return cv2.erode(frame.astype(np.uint8), circular_kernel(kernel_size))


def my_threshold(frame, threshold):
    return frame > threshold


def circular_kernel(kernel_size=3):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


def create_bgr(frame_list, background, thr=0.3):

    bgr_frames = np.zeros((len(frame_list), *background[:, :, 0].shape), dtype=background.dtype)

    for kk in range(len(frame_list)):
        frame = np.asarray(frame_list)[kk, ...]
        # foreground0 = np.abs(background[:, :, 0] - frame[:, :, 0])
        foreground = my_threshold(background[:, :, 0] - frame[:, :, 0], thr * 255)
        foreground = my_erode(foreground, kernel_size=4)
        foreground = cv2.medianBlur(foreground, 3)  # get rid of specks
        bgr_frames[kk, ...] = foreground

    return bgr_frames


def plot_bgrs(vr, iframe, background, thr=0.3, min_y=0, max_y=None, min_x=0, max_x=None):
    plt.rcParams['font.size'] = 16
    iframes = np.arange(iframe-1, iframe+2)
    frame_list = list(vr[iframes.tolist()])
    bgr_frames = create_bgr(frame_list, background, thr=thr)

    plt.figure(figsize=[30, 10])
    for kk in range(bgr_frames.shape[0]):
        plt.subplot(1, 3, kk+1)
        plt.imshow(bgr_frames[kk, ...][min_y:max_y, min_x:max_x])
        plt.axis('off')
        plt.title(iframes[kk])
    plt.savefig(f'f{iframe}_bgr_thr0{int(thr*10)}.png', bbox_inches='tight', transparent=True)


def vplay(frames: np.array, idx: np.array = None):

    import cv2

    if idx is None:
        idx = range(len(frames))

    if len(idx) == len(frames)/2:
        ridx = np.zeros(len(frames), dtype=int)
        ridx[::2], ridx[1::2] = idx, idx
        idx = ridx

    nb_chans = frames.shape[3]

    if nb_chans == 1:
        frames = np.repeat(frames, 3, axis=3)

    cv2.namedWindow('movie', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('movie', (800, 800))

    ii = 0
    while True:
        frame = frames[ii, ...]
        cv2.putText(frame, str(idx[ii]), (12, 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 250), lineType=4)
        cv2.imshow('movie', frame)

        wkey = cv2.waitKey(0)
        if wkey & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        elif wkey & 0xFF == ord('d'):
            ii += 1
        elif wkey & 0xFF == ord('a'):
            ii -= 1

        if ii > len(frames)-1:
            ii = 0
        elif ii < 0:
            ii = len(frames)-1


def rotate_points(x, y, degrees, origin=(0, 0)):
    """Rotate a point around a given point."""
    import math

    radians = degrees / 180 * np.pi
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


def make_bgrframe(background, frame, thr=0.3):

    foreground = my_threshold(background[:, :, 0] - frame[:, :, 0], thr * 255)
    foreground = my_erode(foreground, kernel_size=4)
    foreground = cv2.medianBlur(foreground, 3)  # get rid of specks

    return foreground
