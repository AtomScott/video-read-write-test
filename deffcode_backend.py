from deffcode import FFdecoder
from vidgear.gears import WriteGear
import cv2
import numpy as np
from typing import AnyStr, Tuple, Union, Optional
from logging import getLogger
import json  # Add this import at the top of the file

logger = getLogger("playbox_vision")


class VideoOpenFailure(Exception):
    """Custom exception for video open failures."""
    pass


class VideoStreamDeffcode:
    BACKEND_NAME = "deffcode"

    def __init__(
        self,
        path: AnyStr,
        framerate: Optional[float] = None,
        max_decode_attempts: int = 5,
        verbose: bool = True,
        ffparams: dict = {},
    ):
        """
        Initializes the VideoStreamDeffcode with the specified parameters.

        Args:
            path (str or int): Path to the video file or device index.
            framerate (float, optional): Desired frame rate. If None, uses the video's frame rate.
            max_decode_attempts (int): Maximum attempts to decode a frame.
            frame_format (str): Pixel format for decoded frames.
            verbose (bool): Enable verbose logging.
            **ffparams: Additional FFmpeg parameters.
        """
        if not path:
            raise ValueError("Path must be specified!")
        if framerate is not None and framerate <= 0:
            raise ValueError(f"Specified framerate ({framerate}) is invalid!")
        if max_decode_attempts < 0:
            raise ValueError("Maximum decode attempts must be >= 0!")

        self._path = path
        self._is_device = isinstance(self._path, int)
        self._stream = None
        self._frame_rate = framerate
        self._max_decode_attempts = max_decode_attempts
        self._decode_failures = 0
        self._warning_displayed = False
        self._current_frame_num = 0
        self._frame = None
        self._verbose = verbose
        self._ffparams = ffparams

        self._open_stream()

    def _open_stream(self, framerate: Optional[float] = None):
        try:
            if self._is_device:
                raise VideoOpenFailure("Device streaming is not supported with deffcode.")
            
            self._stream = FFdecoder(
                self._path,
                verbose=self._verbose,
                custom_ffmpeg="/usr/bin/ffmpeg",
                **self._ffparams
            ).formulate()
            self._frame_generator = self._stream.generateFrame()
        except Exception as e:
            raise VideoOpenFailure(f"Failed to open video: {str(e)}")

        if not self._stream:
            raise VideoOpenFailure("Failed to open video stream")

        try:
            metadata_str = self._stream.metadata
            metadata = json.loads(metadata_str)
            if framerate is None:
                framerate = metadata["source_video_framerate"]
                self._frame_rate = framerate
        except json.JSONDecodeError as e:
            raise VideoOpenFailure(f"Failed to decode metadata: {str(e)}")

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def path(self) -> Union[bytes, str]:
        if self._is_device:
            return f"Device {self._path}"
        return self._path

    @property
    def is_seekable(self) -> bool:
        return not self._is_device

    @property
    def frame_size(self) -> Tuple[int, int]:
        if self._frame is not None:
            return (self._frame.shape[1], self._frame.shape[0])
        return (0, 0)  # Default if no frame has been read yet

    @property
    def aspect_ratio(self) -> float:
        if self.frame_size == (0, 0):
            return 1.0
        width, height = self.frame_size
        return width / height if height != 0 else 1.0

    @property
    def position_ms(self) -> float:
        return self._current_frame_num * (1000.0 / self.frame_rate)

    @property
    def frame_number(self) -> int:
        return self._current_frame_num

    def read(self, decode: bool = True) -> Union[np.ndarray, bool]:
        """
        Reads the next frame from the video stream.

        Args:
            decode (bool): If True, returns the decoded RGB frame.
                           If False, returns True if a frame is available.

        Returns:
            np.ndarray or bool: Decoded frame or status.
        """
        if self._stream is None:
            return False

        frame = next(self._frame_generator)
        if frame is None:
            return False
        if decode:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     import os
        #     os.remove("frame.png")
        #     cv2.imwrite("frame.png", frame)
        # exit(0)
        return frame

    def close(self) -> None:
        """
        Closes the video stream and releases resources.
        """
        if self._stream is not None:
            self._stream.terminate()
            self._stream = None
        self._current_frame_num = 0
        self._frame = None
        self._decode_failures = 0

    def reset(self):
        """
        Resets the video stream to the beginning.
        """
        self.close()
        self._open_stream()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class VideoWriterDeffcode:
    BACKEND_NAME = "vidgear_deffcode"

    def __init__(
        self,
        path: str,
        frame_size: Tuple[int, int],
        frame_rate: float,
        codec: str = "h264_nvenc",
        verbose: bool = True,
        **ffparams,
    ):
        """
        Initializes the VideoWriteDeffcode with the specified parameters.

        Args:
            path (str): Path to the output video file.
            frame_size (tuple): Size of the video frames (width, height).
            frame_rate (float): Frame rate of the output video.
            codec (str): Codec to use for encoding.
            verbose (bool): Enable verbose logging.
            **ffparams: Additional FFmpeg parameters.
        """
        if not path:
            raise ValueError("Path must be specified!")
        if frame_rate <= 0:
            raise ValueError(f"Frame rate must be positive, got {frame_rate}")
        if not frame_size or len(frame_size) != 2:
            raise ValueError("Frame size must be a tuple of two integers")

        self.path = path
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.codec = codec
        self.verbose = verbose
        self.ffparams = ffparams

        logger.info(f"Initializing VideoWriteDeffcode with codec: {codec}")
        try:
            output_params = {
                "-input_framerate": self.frame_rate,
                "-vcodec": self.codec,
            }
            # Merge additional FFmpeg parameters
            output_params.update(self.ffparams)

            self.writer = WriteGear(
                output=self.path,
                logging=self.verbose,
                **output_params,
            )
        except Exception as e:
            raise VideoOpenFailure(f"Failed to initialize video writer: {str(e)}")

    def write(self, frame: np.ndarray) -> bool:
        """
        Writes a single frame to the output video.

        Args:
            frame (np.ndarray): BGR frame to write.

        Returns:
            bool: True if successful, False otherwise.
        """
        if frame.shape[:2][::-1] != self.frame_size:
            raise ValueError(
                f"Frame size {frame.shape[:2][::-1]} does not match expected size {self.frame_size}"
            )
        try:
            self.writer.write(frame)
            return True
        except Exception as e:
            logger.error(f"Failed to write frame: {str(e)}")
            return False

    def close(self):
        """
        Finalizes the video file and releases resources.
        """
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()