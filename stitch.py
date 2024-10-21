import yaml
import argparse
import psutil
import os
from stitcher import VideoStitcher  # Import the modified VideoStitcher
def parse_args():
    parser = argparse.ArgumentParser(description="Video Stitching Performance Testing Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def log_cpu_memory():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)  # in MB

def stitch_videos(config):
    input_videos = config["input_videos"]
    output_video = config["output_video"]
    output_params = config.get("output_params", "params.json")
    max_frames = config.get("max_frame_count", -1)
    use_gpu = config.get("use_gpu", False)


    video_stitcher = VideoStitcher({
        "input_videos": input_videos,
        "output_video": output_video,
        "output_params": output_params,
        "use_gpu": use_gpu,
        "debug": config.get("debug", False),
        "stitcher_kwargs": config.get("stitcher_kwargs", {})
    })


    video_stitcher.stitch_videos(
        input_videos=input_videos,
        output_video=output_video,
        output_params=output_params,
        max_frames=max_frames,
    )

def main():
    args = parse_args()
    config = load_config(args.config)

    # Validate input videos
    input_videos = config.get("input_videos", [])
    if not input_videos:
        raise ValueError("No input videos specified in the configuration.")
    for video in input_videos:
        if not os.path.exists(video):
            raise FileNotFoundError(f"Input video does not exist: {video}")

    # Ensure output directory exists
    output_video = config.get("output_video", "output.mp4")
    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stitch_videos(config)

if __name__ == "__main__":
    main()
