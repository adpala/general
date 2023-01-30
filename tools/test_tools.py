import numpy as np
import matplotlib.pyplot as plt


def create_fly(size, position, angle):
    """ Creates a fly of size 'size' pixels, head rotated around thorax by 'angle' degrees clockwise, with thorax in 'position'."""

    # Init
    head_pos = np.zeros((2), dtype=np.float32)
    thorax_pos = np.zeros((2), dtype=np.float32)

    # Rotate
    head_pos[0] = size*np.cos(angle*np.pi/180)
    head_pos[1] = size*np.sin(angle*np.pi/180)

    # Translate
    head_pos += position
    thorax_pos += position

    return head_pos, thorax_pos


def create_nflies(nflies, sizes, positions, angles):
    """ Creates nflies flies of size 'sizes' pixels, head rotated around thorax by 'angles' degrees clockwise, with thorax in 'positions'."""
    # check for dimensions

    # init result variables
    head_pos = np.zeros((nflies, 2), dtype=np.float32)
    thorax_pos = np.zeros((nflies, 2), dtype=np.float32)

    # iterate to create flies
    for ifly in range(nflies):

        ihead_pos, ithorax_pos = create_fly(size=sizes[ifly], position=positions[ifly], angle=angles[ifly])
        head_pos[ifly, ...] = ihead_pos
        thorax_pos[ifly, ...] = ithorax_pos

    return head_pos, thorax_pos


def plot_box(vr, dataset, time, fly=None, mode='cropped'):
    """
    time in seconds
    """
    frame_number = dataset.nearest_frame.loc[time]  # get frame number for that idx
    frame = vr[frame_number]

    if mode == 'full':
        plt.gcf().set_size_inches(30, 15)
        plt.imshow(frame, cmap='Greys')
        plt.plot(dataset.pose_positions_allo.loc[time, :, :, 'x'], dataset.pose_positions_allo.loc[time, :, :, 'y'], '.')
        plt.xlim(0, frame.shape[1])
        plt.ylim(0, frame.shape[0])
    elif mode == 'cropped':
        fly_pos = dataset.pose_positions_allo.loc[time, fly, 'thorax', :].astype(np.uintp)
        print(fly_pos)
        box_size = np.uintp(100)
        x_range = np.clip((fly_pos.loc['x']-box_size, fly_pos.loc['x']+box_size), 0, vr.frame_width-1)
        y_range = np.clip((fly_pos.loc['y']-box_size, fly_pos.loc['y']+box_size), 0, vr.frame_height-1)
        plt.imshow(frame[slice(*y_range), slice(*x_range), :], cmap='Greys')
    plt.title(f"time {time}")
    plt.show()


def plot_boxes(vr, dataset, frame_numbers, nflies):
    """
    time in seconds
    """
    frames = list(vr[frame_numbers.tolist()])
    times = [dataset.time[dataset.nearest_frame == ff][0].values for ff in frame_numbers]

    for ff, frame in enumerate(frames):
        plt.figure(figsize=[10, 10])
        plt.imshow(frame, cmap='Greys')
        plt.plot(dataset.pose_positions_allo.loc[times[ff], :, :, 'x'], dataset.pose_positions_allo.loc[times[ff], :, :, 'y'], '.')
        for ifly in range(nflies):
            plt.text(dataset.pose_positions_allo.loc[times[ff], ifly, 'thorax', 'x']+30, dataset.pose_positions_allo.loc[times[ff], ifly, 'thorax', 'y']-30, f"{ifly}", color='red', weight='bold')
        plt.xlim(0, frame.shape[1])
        # plt.ylim(0, frame.shape[0])
        plt.title(f"time {times[ff]}")
        plt.axis('off')
        plt.show()
