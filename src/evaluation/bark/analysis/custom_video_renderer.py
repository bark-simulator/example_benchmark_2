import logging
import os

from bark.runtime.viewer.video_renderer import VideoRenderer

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import cm

class CustomVideoRenderer(VideoRenderer):
  def __init__(self, **kwargs):
    super(CustomVideoRenderer, self).__init__(renderer = None, **kwargs)

  def DumpFrame(self, figure):
    image_path = os.path.join(self.video_frame_dir, "{:03d}.png".format(self.frame_count))
    figure.savefig(image_path)
    self.frame_count = self.frame_count + 1