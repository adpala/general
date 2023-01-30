from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlFile
from pyforms.controls import ControlPlayer
import deepdish as dd
import numpy as np
import cv2


class TrackInspector(BaseWidget):

    def __init__(self, *args, **kwargs):
        super().__init__('Inspector Track')

        datename = 'localhost-20190904_151239'

        root_name = 'Z:/#Common/backlight'
        res_dir = 'res'
        dat_dir = 'dat'

        # Definition of the forms fields
        self._videofile = ControlFile('Video')
        self._videofile.value = f'{root_name}/{dat_dir}/{datename}/{datename}.avi'
        self._player = ControlPlayer('Player')

        # Define the function that will be called when a file is selected
        self._videofile.changed_event = self.__videoFileSelectionEvent
        self._player.process_frame_event = self.__process_frame

        # Define the organization of the Form Controls
        self._formset = ['_videofile', '_player']

        filename = f'{root_name}/{res_dir}/{datename}/{datename}_tracks.h5'
        self.x, body_parts, first_tracked_frame, last_tracked_frame = self.load_tracks(filename)
        self.nb_flies = self.x.shape[1]
        self.trail_hist = 50
        self.colors = self.make_colors(self.nb_flies)

    def load_tracks(self, filename):
        data = dd.io.load(filename)
        chbb = data['chambers_bounding_box'][:]
        heads = data['lines'][:, 0, :, 0, ::-1]   # nframe, fly id, coordinates
        tails = data['lines'][:, 0, :, 1, ::-1]   # nframe, fly id, coordinates
        box_centers = data['centers'][:, 0, :, :]   # nframe, fly id, coordinates
        # everything to frame coords
        heads = heads + chbb[1][0][:]
        tails = tails + chbb[1][0][:]
        box_centers = box_centers + chbb[1][0][:]
        body_parts = ['head', 'center', 'tail']
        first_tracked_frame, last_tracked_frame = data['start_frame'], data['frame_count']
        x = np.stack((heads, box_centers, tails), axis=2).astype(np.uintp)
        # x = x[first_tracked_frame:last_tracked_frame, ...]
        return x, body_parts, first_tracked_frame, last_tracked_frame

    def make_colors(self, nb_flies):
        colors = np.zeros((1, nb_flies, 3), np.uint8)
        colors[0, :, 1:] = 200  # set saturation and brightness to 220
        colors[0, :, 0] = np.arange(0, 180, 180.0 / nb_flies)  # set range of hues
        colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0]
        colors = [thisColor.tolist() for thisColor in colors]  # convert all items in color list to float
        return colors

    def __videoFileSelectionEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self._player.value = self._videofile.value

    def __process_frame(self, frame):
        """
        Do some processing to the frame and return the result frame
        """
        idx = self._player.video_index
        for fly in range(self.nb_flies):
            y, x = self.x[idx, fly, 2, :]
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            col = self.colors[fly]
            for ii in range(max(0, idx - self.trail_hist), idx):
                y, x = self.x[ii, fly, 1, :]
                cv2.circle(frame, (x, y), 2, col, -1)
        return frame


if __name__ == '__main__':

    from pyforms import start_app
    start_app(TrackInspector)
