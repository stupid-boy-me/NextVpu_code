import os, shutil
from pathlib import Path


def mkdir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)


def rmdir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def get_video_paths(root_folder, extensions=None):
    if extensions is None:
        extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".flv",
            ".wmv",
            ".webm",
            ".ts",
            ".h265",
        }
    root_path = Path(root_folder)
    video_paths = []
    for file_path in root_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            video_paths.append(str(file_path.absolute()))
    return video_paths
