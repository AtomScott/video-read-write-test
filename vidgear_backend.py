from vidgear.gears import WriteGear, CamGear
import numpy as np

from typing import AnyStr, Tuple, Union, Optional
from logging import getLogger

logger = getLogger("playbox_vision")

class VideoStreamVidgear():
    BACKEND_NAME = "vidgear"

    def __init__(
        self,
        path: AnyStr = None,
        framerate: Optional[float] = None,
        max_decode_attempts: int = 5,
        **kwargs,
    ):
        super().__init__()
        if path is None:
            raise ValueError("Path must be specified!")
        if framerate is not None:
            raise ValueError(f"Specified framerate ({framerate}) is invalid!")
        if max_decode_attempts < 0:
            raise ValueError("Maximum decode attempts must be >= 0!")

        self._path = path
        self._is_device = isinstance(self._path, int)
        self._stream = None
        self._frame_rate = None
        self._max_decode_attempts = max_decode_attempts
        self._decode_failures = 0
        self._warning_displayed = False
        self._current_frame_num = 0
        self._frame = None

        self._open_stream(framerate, **kwargs)

    def _open_stream(self, framerate: Optional[float] = None, **kwargs):
        try:
            self._stream = CamGear(source=self._path, **kwargs).start()
        except Exception as e:
            raise VideoOpenFailure(f"Failed to open video: {str(e)}")

        if not self._stream:
            raise VideoOpenFailure("Failed to open video stream")

        if framerate is None:
            framerate = self._stream.framerate

        self._frame_rate = framerate

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
        return (0, 0)  # Return default if no frame has been read yet

    @property
    def aspect_ratio(self) -> float:
        # Assuming square pixels as VidGear doesn't provide this information
        return 1.0

    @property
    def position_ms(self) -> float:
        return self._current_frame_num * (1000.0 / self.frame_rate)

    @property
    def frame_number(self) -> int:
        return self._current_frame_num

    def read(
        self, decode: bool = True, advance: bool = True
    ) -> Union[np.ndarray, bool]:
        if self._stream is None:
            return False

        if advance:
            for _ in range(self._max_decode_attempts):
                self._frame = self._stream.read()
                if self._frame is not None:
                    self._current_frame_num += 1
                    return self._frame if decode else True

            self._decode_failures += 1
            if not self._warning_displayed and self._decode_failures > 1:
                logger.warning(
                    "Failed to decode some frames, results may be inaccurate."
                )
                self._warning_displayed = True
            return False

        if decode and self._frame is not None:
            return self._frame
        return self._frame is not None

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream = None
        self._current_frame_num = 0
        self._frame = None
        self._decode_failures = 0

    def reset(self):
        self.close()
        self._open_stream(self._frame_rate)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class VideoWriterVidgear():
    BACKEND_NAME = "vidgear"

    def __init__(
        self, path: str, frame_size: tuple, frame_rate: float, codec: str = "libx264"
    ):
        self.path = path
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.codec = codec
        logger.info(f"Initializing VideoWriterVidgear with codec: {codec}")
        
        output_params = {
            "-input_framerate": frame_rate,
            "-vcodec": codec,
        }

        self.writer = WriteGear(
            output=self.path,
            compression_mode=True,
            logging=True,
            custom_ffmpeg='/usr/bin/ffmpeg',
            **output_params,
        )

    def write(self, frame: np.ndarray) -> bool:
        if frame.shape[:2][::-1] != self.frame_size:
            raise ValueError(
                f"Frame size {frame.shape[:2][::-1]} does not match expected size {self.frame_size}"
            )

        self.writer.write(frame)
        return True

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()