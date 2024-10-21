from logging import getLogger
import os
from typing import List, Optional, Generator, Tuple

import cv2
import json
import numpy as np
from stitching import Stitcher
from stitching.images import Images
from tqdm import tqdm
from vidgear_backend import VideoStreamVidgear, VideoWriterVidgear
from deffcode_backend import VideoStreamDeffcode, VideoWriterDeffcode

logger = getLogger("playbox_vision")

def open_video(video_path, use_gpu):
    if use_gpu:
        ffparams = {
            "-vcodec": None,  # skip source decoder and let FFmpeg chose
            "-enforce_cv_patch": True, # enable OpenCV patch for YUV(NV12) frames
            "-ffprefixes": [
                "-vsync",
                "0",  # prevent duplicate frames
                "-hwaccel",
                "cuda",  # accelerator
                "-hwaccel_output_format",
                "cuda",  # output accelerator
            ],
            "-custom_resolution": "null",  # discard source `-custom_resolution`
            "-framerate": "null",  # discard source `-framerate`
            "-vf": "hwdownload,format=nv12,format=rgb24"  # scale to 640x360 in GPU memory
        }
        return VideoStreamDeffcode(video_path, ffparams=ffparams)
    else:
        return VideoStreamVidgear(video_path)

def write_video(video_path, frame_size, frame_rate, use_gpu):
    if use_gpu:
        return VideoWriterVidgear(video_path, frame_size, frame_rate, codec="h264_nvenc")
    else:
        return VideoWriterVidgear(video_path, frame_size, frame_rate)

class VideoStitcher(Stitcher):
    DEFAULT_STITCHER_SETTINGS = Stitcher.DEFAULT_SETTINGS
    DEFAULT_VIDEO_STITCHER_SETTINGS = {
        "video_length": -1,
        "backend": "moviepy",
        "input_videos": [],
        "output_video": "",
        "output_params": "",
        "stitcher_kwargs": {},
        "debug": False,
        "use_gpu": False,
    }

    def __init__(self, config={}):
        config = {**self.DEFAULT_VIDEO_STITCHER_SETTINGS, **config}
        Stitcher.__init__(
            self, **{**self.DEFAULT_STITCHER_SETTINGS, **config["stitcher_kwargs"]}
        )
        self.input_videos = config["input_videos"]
        self.output_video = config["output_video"]
        self.output_params = config["output_params"]
        self.params = {}
        self.params_saved = False
        self.cameras = None
        self.cameras_registered = False
        self.backend = config["backend"]
        self.video_streams = None
        self.video_writer = None
        self.video_length = config["video_length"]
        self.warper_type = "spherical"
        self.debug = config["debug"]
        self.use_gpu = config.get("use_gpu", False)
        self.cuda_available = self.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_gpu and not self.cuda_available:
            logger.warning("CUDA is not available. Falling back to CPU.")
            self.use_gpu = False

    def stitch_videos(
        self,
        input_videos: List[str],
        output_video: Optional[str] = None,
        output_params: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        """Stitch multiple input videos into a single panoramic video.

        Args:
            input_videos: List of paths to input video files.
            output_video: Path to the output stitched video file.
            output_params: Path to the output stitched video parameters file.
            max_frames: Maximum number of frames to process (optional).
        """
        self.video_streams = [open_video(video, self.use_gpu) for video in input_videos]
        frame_rate = self.video_streams[0].frame_rate

        for frame in tqdm(self._stitch_frame_by_frame(max_frames), total=max_frames):
            if self.video_writer is None:
                self.video_writer = self._initialize_video_writer(
                    frame, output_video or self.output_video, frame_rate
                )  # Just to shut up mypy
            self.video_writer.write(frame)
            if (not self.params_saved) and self.cameras_registered:
                with open(output_params or self.output_params, "w") as f:
                    json.dump(self.params, f, indent=4)
                self.params_saved = True
        logger.info(f"Video stitching completed. Output saved to: {output_video}")
        self._cleanup()

    def stitch_frame_by_frame(
        self, input_videos: List[str], max_frames: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        """Generator that yields stitched frames without writing to a file.

        Args:
            input_videos: List of paths to input video files.
            max_frames: Maximum number of frames to process (optional).

        Yields:
            Stitched panoramic frames as numpy arrays.
        """
        self.video_streams = [open_video(video, self.use_gpu) for video in input_videos]
        yield from self._stitch_frame_by_frame(max_frames)
        self._cleanup()

    def stitch(self, frames):
        return self._stitch_single_frame(frames)

    def stitch_frames(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, dict]:
        """Stitch multiple input frames into a single panoramic frame.

        Args:
            frames: List of input frames as numpy arrays.

        Returns:
            Stitched panoramic frame as a numpy array.
            Params saved to self.params
        """
        return self._stitch_single_frame(frames), self.params

    def register_cameras(self, feature_masks=[]):
        # Medium resolution processing
        medium_res_imgs = self.resize_medium_resolution()
        features = self.find_features(medium_res_imgs, feature_masks)
        matches = self.match_features(features)
        subset_imgs, subset_features, subset_matches = self.subset(
            medium_res_imgs, features, matches
        )

        # Camera parameter estimation
        initial_cameras = self.estimate_camera_parameters(
            subset_features, subset_matches
        )
        refined_cameras = self.refine_camera_parameters(
            subset_features, subset_matches, initial_cameras
        )
        wave_corrected_cameras = self.perform_wave_correction(refined_cameras)
        self.estimate_scale(wave_corrected_cameras)
        self.cameras = wave_corrected_cameras

        # Low resolution processing
        low_res_imgs = self.resize_low_resolution(subset_imgs)
        warped_low_res = self.warp_low_resolution(low_res_imgs, self.cameras)
        low_res_imgs, low_res_masks, low_res_corners, low_res_sizes = warped_low_res

        self.prepare_cropper(
            low_res_imgs, low_res_masks, low_res_corners, low_res_sizes
        )
        cropped_low_res = self.crop_low_resolution(
            low_res_imgs, low_res_masks, low_res_corners, low_res_sizes
        )
        cropped_imgs, cropped_masks, cropped_corners, cropped_sizes = cropped_low_res

        self.estimate_exposure_errors(cropped_corners, cropped_imgs, cropped_masks)
        seam_masks = self.find_seam_masks(cropped_imgs, cropped_corners, cropped_masks)
        self.seam_masks = list(seam_masks)

        # Final resolution processing
        final_res_imgs = list(self.resize_final_resolution())
        warped_final_res = self.warp_final_resolution(final_res_imgs, self.cameras)
        _, final_masks, final_corners, final_sizes = warped_final_res

        self.masks = list(final_masks)
        self.corners = list(final_corners)
        self.sizes = list(final_sizes)

        self.src_sizes_hw = [img.shape[:2] for img in final_res_imgs]
        self.dst_sizes_hw = [(size[1], size[0]) for size in final_sizes]

        self.uxmaps, self.vymaps, self.uvbounds = self.build_maps(
            self.cameras, self.src_sizes_hw, self.dst_sizes_hw
        )

        self.cameras_registered = True
        if self.debug:
            output_dir = os.path.dirname(self.output_video)
            os.makedirs(f"{output_dir}/debug", exist_ok=True)
            for i, img in enumerate(final_res_imgs):
                cv2.imwrite(f"{output_dir}/debug/register_cameras-image_{i}.jpg", img)
                cv2.imwrite(
                    f"{output_dir}/debug/register_cameras-mask_{i}.jpg", self.masks[i]
                )
                cv2.imwrite(
                    f"{output_dir}/debug/register_cameras-seam_{i}.jpg",
                    self.seam_masks[i],
                )

    def _stitch_frame_by_frame(
        self, max_frames: Optional[int]
    ) -> Generator[np.ndarray, None, None]:
        frame_count = 0
        while True:
            frame_set = self._read_frames()
            if not frame_set:
                break

            stitched_frame = self._stitch_single_frame(frame_set)
            if stitched_frame is None:
                raise RuntimeError("Error stitching frames")

            yield stitched_frame

            frame_count += 1
            if max_frames < 0:
                continue
            if frame_count >= max_frames:
                break

    def _read_frames(self) -> List[np.ndarray]:
        frame_set = []
        for video in self.video_streams:
            frame = video.read()
            if frame is False:
                return []
            frame_set.append(frame)
        return frame_set

    def _stitch_single_frame(self, images: List[np.ndarray]) -> np.ndarray:
        self.images = Images.of(
            images,
            self.medium_megapix,
            self.low_megapix,
            self.final_megapix,
        )

        if not self.cameras_registered:
            self.register_cameras()

        imgs = self.resize_final_resolution()
        
        # Use GPU accelerated warping if enabled
        if self.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            imgs = self.warp_images_gpu(imgs, self.uxmaps, self.vymaps)
        else:
            imgs = self.warp_images(imgs, self.uxmaps, self.vymaps)

        imgs, masks, corners, sizes = self.crop_final_resolution(
            imgs,
            self.masks,
            self.corners,
            self.sizes,
        )

        tmp_masks = self.masks  # FIXME: Horrible but blending uses masks implicitly
        self.masks = list(masks)
        imgs = list(self.compensate_exposure_errors(corners, imgs))
        seam_masks = list(self.resize_seam_masks(self.seam_masks))

        self.initialize_composition(corners, sizes)
        self.blend_images(imgs, seam_masks, corners)
        panorama, _ = self.blender.blend()
        self.masks = tmp_masks

        return panorama

    def _initialize_video_writer(
        self, frame: np.ndarray, output_path: str, frame_rate: float
    ):
        frame_size = (frame.shape[1], frame.shape[0])
        self.video_writer = write_video(
            output_path, frame_size, frame_rate, self.use_gpu
        )
        return self.video_writer

    def _cleanup(self):
        if self.video_streams:
            for stream in self.video_streams:
                stream.close()
        if self.video_writer:
            self.video_writer.close()
        self.video_streams = None
        self.video_writer = None

    def set_masks(self, mask_generator):
        self.masks = list(mask_generator)

    def get_mask(self, idx):
        return self.masks[idx]

    def get_uvbounds(self):
        uvb_min = np.min(self.uvbounds, axis=0)
        uvb_max = np.max(self.uvbounds, axis=0)

        uvbounds = np.array(
            [uvb_min[0, 0], uvb_max[0, 1], uvb_min[1, 0], uvb_max[1, 1]]
        )
        return uvbounds

    def build_maps(self, cameras, src_sizes, dst_sizes):
        aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.FINAL
        )
        uxmaps, vymaps, uvbounds = [], [], []
        self.params["cameras"] = {}
        self.params["src_sizes"] = {}
        self.params["dst_sizes"] = {}
        self.params["uvbounds"] = {}
        for i, (camera, src_size, dst_size) in enumerate(
            zip(cameras, src_sizes, dst_sizes)
        ):
            K = self.get_K(camera, aspect)
            R = camera.R
            uxmap, vymap, uvbound = self.build_map(K, R, src_size, dst_size)
            uxmaps.append(uxmap)
            vymaps.append(vymap)
            uvbounds.append(uvbound)
            self.params["cameras"][i] = {
                "K": K.tolist(),
                "R": R.tolist(),
            }
            self.params["src_sizes"][i] = src_size
            self.params["dst_sizes"][i] = dst_size
            self.params["uvbounds"][i] = uvbound

        return uxmaps, vymaps, uvbounds

    def build_map(self, K, R, src_size, dst_size):
        src_h, src_w = src_size
        dst_h, dst_w = dst_size

        points_3d = create_3d_points(src_h, src_w)
        rotated_points = apply_rotation(points_3d, K, R)
        X, Y, Z = (
            rotated_points[:, :, 0],
            rotated_points[:, :, 1],
            rotated_points[:, :, 2],
        )
        r = np.sqrt(X**2 + Y**2 + Z**2)
        u = np.arctan2(X, Z)
        v = np.pi - np.arccos(np.clip(Y / r, -1, 1))
        uvbound = [(u.min(), u.max()), (v.min(), v.max())]

        u, v = np.meshgrid(
            np.linspace(*uvbound[0], dst_w),
            np.linspace(*uvbound[1], dst_h),
        )

        sinv = np.sin(np.pi - v)
        points_3d = np.stack(
            (sinv * np.sin(u), np.cos(np.pi - v), sinv * np.cos(u)), axis=-1
        )
        rotated_points = points_3d @ np.linalg.inv(R.T) @ K.T
        x, y, z = (
            rotated_points[:, :, 0],
            rotated_points[:, :, 1],
            rotated_points[:, :, 2],
        )
        uxmap, vymap = x / z, y / z
        return uxmap.astype(np.float32), vymap.astype(np.float32), uvbound

    def warp_images(self, imgs, uxmaps, vymaps):
        return [
            cv2.remap(
                img,
                uxmap,
                vymap,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            for img, uxmap, vymap in zip(imgs, uxmaps, vymaps)
        ]

    def warp_images_gpu(self, imgs, uxmaps, vymaps):
        """Warp images using GPU accelerated remap."""
        warped_imgs = []
        try:
            for img, uxmap, vymap in zip(imgs, uxmaps, vymaps):
                gpu_img = cv2.cuda_GpuMat()
                gpu_uxmap = cv2.cuda_GpuMat()
                gpu_vymap = cv2.cuda_GpuMat()

                gpu_img.upload(img)
                gpu_uxmap.upload(uxmap)
                gpu_vymap.upload(vymap)

                gpu_warped = cv2.cuda.remap(
                    gpu_img,
                    gpu_uxmap,
                    gpu_vymap,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )

                warped_img = gpu_warped.download()
                warped_imgs.append(warped_img)
        except cv2.error as e:
            logger.error(f"GPU warping failed: {e}. Falling back to CPU warping.")
            self.use_gpu = False
            warped_imgs = self.warp_images(imgs, uxmaps, vymaps)

        return warped_imgs

    def get_K(self, camera, aspect=1):
        K = camera.K().astype(np.float32)
        """ Modification of intrinsic parameters needed if cameras were
        obtained on different scale than the scale of the Images which should
        be warped """
        K[0, 0] *= aspect
        K[0, 2] *= aspect
        K[1, 1] *= aspect
        K[1, 2] *= aspect
        return K


def create_3d_points(h, w):
    y, x = np.indices((h, w))
    return np.stack((x, y, np.ones_like(x)), axis=-1)


def apply_rotation(points_3d, K, R):
    return points_3d @ np.linalg.inv(K.T) @ R.T