#!/usr/bin/env python3

import argparse
import os

import cv2


def main(args: argparse.Namespace):
    figures = [item for item in os.listdir(args.input_dir) if item.endswith(".png")]
    figures = sorted(figures, key=lambda x: int(x.split("_")[-1].split(".")[0]))  # Sort figures by idx

    assert args.output_video.endswith(".mp4"), "This script can only generate mp4"

    first_image = cv2.imread(figures[0])
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, (width, height))

    for image_path in figures:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping...")
            continue

        if (image.shape[1], image.shape[0]) != (width, height):
            image = cv2.resize(image, (width, height))

        video_writer.write(image)

    video_writer.release()
    print(f"Video saved as {args.output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to folder containing generated plots")
    parser.add_argument("--output_video", type=str, default="generated.mp4", help="Name of the generated video")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    main(args)
