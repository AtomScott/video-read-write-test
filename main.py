import cv2
import time
import yaml
import argparse
import psutil
import os
from tqdm import tqdm

try:
    import cupy as cp
    import cv2.cuda

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Video Performance Testing Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log_cpu_memory():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)  # in MB


def process_frame_cpu(frame):
    # Simulate processing (e.g., resizing)
    processed_frame = cv2.resize(frame, (1920, 1080))
    return processed_frame


def process_frame_gpu(frame):
    # Upload to GPU memory
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    # Simulate processing (e.g., resizing)
    gpu_frame = cv2.cuda.resize(gpu_frame, (1920, 1080))
    # Download back to CPU memory
    processed_frame = gpu_frame.download()
    return processed_frame


def read_write_video(config, input_file, output_file):
    total_video_reading_time = 0.0
    total_video_writing_time = 0.0
    total_frame_reading_time = 0.0
    total_frame_writing_time = 0.0
    frame_count = 0

    cap = cv2.VideoCapture(input_file)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Respect max_frame_count from config
    max_frames = min(total_frames, config.get("max_frame_count", total_frames))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_processing_times = []
    memory_usages = []

    for _ in tqdm(range(max_frames), desc="Processing frames"):
        # Video Reading
        start = time.time()
        ret, frame = cap.read()
        end = time.time()
        total_video_reading_time += end - start

        if not ret:
            break

        frame_count += 1
        start_time = time.time()

        if config["use_gpu"]:
            if not CUDA_AVAILABLE:
                raise Exception("CUDA is not available or OpenCV is not compiled with CUDA support.")
            processed_frame = process_frame_gpu(frame)
        else:
            processed_frame = process_frame_cpu(frame)

        # Video Writing: Opening VideoWriter
        start = time.time()
        out.write(processed_frame)
        end = time.time()
        total_video_writing_time += end - start

        # Frame Writing
        start = time.time()
        cv2.imwrite("frame.png", processed_frame)
        end = time.time()
        total_frame_writing_time += end - start

        # Frame Reading
        start = time.time()
        frame = cv2.imread("frame.png")
        end = time.time()
        total_frame_reading_time += end - start

        os.remove("frame.png")

        end_time = time.time()
        frame_processing_times.append(end_time - start_time)

        # Log memory usage
        cpu_memory = log_cpu_memory()
        memory_usage = {"cpu_memory_mb": cpu_memory}

        memory_usages.append(memory_usage)

    cap.release()
    out.release()

    task_times = {
        "avg_video_reading_time": total_video_reading_time / frame_count,
        "avg_video_writing_time": total_video_writing_time / frame_count,
        "avg_frame_reading_time": total_frame_reading_time / frame_count,
        "avg_frame_writing_time": total_frame_writing_time / frame_count,
    }

    return frame_processing_times, memory_usages, task_times


def main():
    args = parse_args()
    config = load_config(args.config)

    start_time = time.time()

    frame_times, memory_usages, task_times = read_write_video(config, config["input_file"], config["output_file"])
    total_time = time.time() - start_time

    print(f"Metrics for {config['input_file']}")
    print(f"Total Processing Time: {total_time:.2f} seconds")
    print(f"Average Frame Processing Time: {sum(frame_times)/len(frame_times):.4f} seconds")

    if memory_usages:
        avg_cpu_memory = sum([m["cpu_memory_mb"] for m in memory_usages]) / len(memory_usages)
        print(f"Average CPU Memory Usage: {avg_cpu_memory:.2f} MB")

    # Output Task Times
    print("\nTask Timings:")
    print(f"task: avg_video_reading time: {task_times['avg_video_reading_time']:.4f} seconds")
    print(f"task: avg_video_writing time: {task_times['avg_video_writing_time']:.4f} seconds")
    print(f"task: avg_frame_reading time: {task_times['avg_frame_reading_time']:.4f} seconds")
    print(f"task: avg_frame_writing time: {task_times['avg_frame_writing_time']:.4f} seconds")


if __name__ == "__main__":
    main()
