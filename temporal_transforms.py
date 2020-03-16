import random
import math
import numpy as np


class LoopPadding(object):

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample
        out = frame_indices

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample

        out = frame_indices[:clip_duration]

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return selected_frames


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (clip_duration // 2))
        end_index = min(begin_index + clip_duration, vid_duration)

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return selected_frames


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        vid_duration  = len(frame_indices)
        clip_duration = self.size * self.downsample

        rand_end = max(0, vid_duration - clip_duration - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + clip_duration, vid_duration)

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= clip_duration:
                break
            out.append(index)

        selected_frames = [out[i] for i in range(0, clip_duration, self.downsample)]

        return selected_frames


class TemporalSelectCrop(object):
    def __init__(self, size, downsample, number_clips=6, clip_interval=0):
        self.size = size
        self.downsample = downsample
        self.clip_duration = self.size * self.downsample
        self.number_clips = number_clips
        self.clip_interval = clip_interval

    def __call__(self, frame_indices):
        vid_duration = len(frame_indices)

        outs = []
        if (self.clip_duration + self.clip_interval) * self.number_clips - self.clip_interval <= vid_duration:
            center_index = len(frame_indices) // 2
            begin_index = max(0, center_index -
                              (self.clip_duration // 2 +
                               self.number_clips // 2 * (self.clip_duration + self.clip_interval)))
            for i in range(self.number_clips):
                end_index = min(begin_index + self.clip_duration, vid_duration)
                out = frame_indices[begin_index:end_index]
                outs.append(out)
                begin_index = end_index + self.clip_interval
        else:
            for i in range(self.number_clips):
                rand_end = max(0, vid_duration - self.clip_duration - 1)
                begin_index = random.randint(0, rand_end)
                end_index = min(begin_index + self.clip_duration, vid_duration)
                out = frame_indices[begin_index:end_index]
                outs.append(out)

        total_frames = []
        for out in outs:
            for index in out:
                if len(out) >= self.clip_duration:
                    break
                out.append(index)
            frames = [out[i] for i in range(0, self.clip_duration, self.downsample)]
            total_frames.append(frames)

        return total_frames


class TemporalBeginEndCrop(object):
    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample
        self.clip_duration = self.size * self.downsample

    def __call__(self, frame_indices):
        vid_duration = len(frame_indices)

        outs = []
        begin = frame_indices[:self.clip_duration]
        end = frame_indices[-self.clip_duration:]
        outs.append(begin)
        outs.append(end)

        total_frames = []
        for out in outs:
            for index in out:
                if len(out) >= self.clip_duration:
                    break
                out.append(index)
            frames = [out[i] for i in range(0, self.clip_duration, self.downsample)]
            total_frames.append(frames)

        return total_frames


class TemporalRandomMultipleCrop(object):
    '''Crop out temporal randomly

    size: desired output temporal length
    number_clips: number of clips for each video
    clip_interval: minimal distance between different clips, if -1 then simply randomly
    '''
    def __init__(self, size, downsample, number_clips=4, clip_interval=-1):
        self.size = size
        self.downsample = downsample
        # self.clip_duration = self.size * self.downsample
        self.number_clips = number_clips
        self.clip_interval = clip_interval

    def _randomCrop(self, frame_indices):
        vid_duration = len(frame_indices)
        rand_end = max(0, vid_duration - self.clip_duration)
        begin_index = random.randint(0, rand_end)  # [0, rand_end]
        # it might be shorter than desired, than same frames will be appended to fill
        end_index = min(begin_index + self.clip_duration, vid_duration)

        return frame_indices[begin_index : end_index]

    def __call__(self, frame_indices):
        vid_duration = len(frame_indices)  # length of sample video
        downsample = random.randint(1, self.downsample)
        self.clip_duration = self.size * downsample

        outs = []
        if self.clip_interval < 0:
            # randomly choose clips
            outs = [self._randomCrop(frame_indices) for _ in range(self.number_clips)]
        else:
            rand_begin = vid_duration - self.clip_duration * (self.number_clips-1) - self.clip_interval * (self.number_clips-1)
            if rand_begin < 0:
                begin_index = random.randint(0, rand_begin)
            else:
                begin_index = 0
            for i in range(self.number_clips):
                end_index = min(begin_index + self.clip_duration, vid_duration)
                out = frame_indices[begin_index : end_index]
                outs.append(out)
                begin_index = min(begin_index)

        total_frames = []
        for out in outs:
            for index in out:
                if len(out) >= self.clip_duration:
                    break
                out.append(index)
            frames = [out[i] for i in range(0, self.clip_duration, downsample)]
            total_frames.append(frames)

        return total_frames


class TemporalSequentialCrop(object):
    """Crop out a list of clips which cover the whole video"""
    def __init__(self, size, downsample, number_clips=4):
        self.size = size
        self.downsample = downsample
        self.number_clips = number_clips
        self.clip_duration = self.size * self.downsample

    def _randomCrop(self, snippet):
        """random crop out a sublist from snippet (a list)"""
        rand_end = max(0, len(snippet) - self.clip_duration)
        begin_index = random.randint(0, rand_end)

        return snippet[begin_index:begin_index + self.clip_duration]

    def __call__(self, frame_indices):
        vid_duration = len(frame_indices)

        clips = []
        if vid_duration >= self.clip_duration * self.number_clips:
            snippet_interval = vid_duration // self.number_clips
            break_pts = [i * snippet_interval for i in range(self.number_clips)]
            for pt in break_pts:
                clips.append(self._randomCrop(frame_indices[pt:min(pt + self.clip_duration, vid_duration)]))
        else:
            snippet_num = vid_duration // self.clip_duration
            break_pts = [i * self.clip_duration for i in range(snippet_num)]
            for i, pt in enumerate(break_pts):
                if i == snippet_num - 1:
                    clip = frame_indices[pt:]
                    clips.append(self._randomCrop(clip))
                else:
                    clips.append(frame_indices[pt:pt + self.clip_duration])
            for i in range(len(clips), self.number_clips):
                clips.append(self._randomCrop(frame_indices))

        assert len(clips) == self.number_clips, 'Fatal Error!'

        total_frames = []
        for clip in clips:
            frames = [clip[i] for i in range(0, self.clip_duration, self.downsample)]
            total_frames.append(frames)

        return total_frames


if __name__ == '__main__':
    # temporal_transform = TemporalSelectCrop(16, 1, number_clips=6, clip_interval=20)
    # temporal_transform = TemporalBeginEndCrop(16, 1)
    # temporal_transform = TemporalRandomMultipleCrop(16, 1)
    temporal_transform = TemporalSequentialCrop(16, 1, number_clips=8)

    # use a dummy frame indices (25 FPS for 10s video)
    frame_indices = list(range(60))
    total_frames = temporal_transform(frame_indices)
    print('Has {} frames in total'.format(len(total_frames)))

    for frames in total_frames:
        print(frames)
