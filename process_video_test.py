import json

# Path to the JSON file
json_path = '../ComfyUI/input/NagArjuna.json'

# Load the JSON data
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract paths for frames 144 to 221 (inclusive)
frame_paths = [
    "../ComfyUI/" + data[str(frame_number)]["original_frame"]
    for frame_number in range(144, 222)
    if str(frame_number) in data
]

print(len(frame_paths), frame_paths[0], frame_paths[-1])

source_paths = ["../ComfyUI/input/Sharukh.jpg"]

from facefusion.processors.modules.face_swapper import process_video
import time

print(help(process_video))

start_time = time.time()
process_video(source_paths, frame_paths)
print(time.time() - start_time)