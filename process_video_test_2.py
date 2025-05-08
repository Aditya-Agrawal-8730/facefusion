import json
import time

from facefusion.program import create_program
from facefusion.processors.core import load_processor_module
from facefusion import state_manager
from facefusion.args import apply_args

json_path = '../ComfyUI/input/NagArjuna.json'
with open(json_path, 'r') as f:
    data = json.load(f)

frame_paths = ["../ComfyUI/" + data[str(frame_number)]["original_frame"] for frame_number in range(144, 222) if str(frame_number) in data]

print(len(frame_paths), frame_paths[0], frame_paths[-1])

source_paths = ["../ComfyUI/input/Sharukh.jpg"]

program = create_program()
args = vars(program.parse_args())
apply_args(args, state_manager.init_item)

processor_module = load_processor_module("face_swapper")
# print(help(processor_module))

start_time = time.time()
processor_module.process_video(source_paths, frame_paths)
print(time.time() - start_time)