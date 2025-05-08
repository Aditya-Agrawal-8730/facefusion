import cv2
import os
import json
import shutil
from datetime import datetime
import time

from facefusion.processors.core import load_processor_module
from facefusion.core import conditional_append_reference_faces

from facefusion import cli_helper, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, logger, process_manager, state_manager, voice_extractor, wording
from facefusion.args import apply_args, collect_job_args, reduce_job_args, reduce_step_args
from facefusion.common_helper import get_first
from facefusion.content_analyser import analyse_image, analyse_video
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.exit_helper import graceful_exit, hard_exit
from facefusion.face_analyser import get_average_face, get_many_faces, get_one_face
from facefusion.face_selector import sort_and_filter_faces
from facefusion.face_store import append_reference_face, clear_reference_faces, get_reference_faces
from facefusion.ffmpeg import copy_image, extract_frames, finalize_image, merge_video, replace_audio, restore_audio
from facefusion.filesystem import filter_audio_paths, get_file_name, is_image, is_video, resolve_file_paths, resolve_file_pattern
from facefusion.jobs import job_helper, job_manager, job_runner
from facefusion.jobs.job_list import compose_job_list
from facefusion.memory import limit_system_memory
from facefusion.processors.core import get_processors_modules
from facefusion.program import create_program
from facefusion.program_helper import validate_args
from facefusion.statistics import conditional_log_statistics
from facefusion.temp_helper import clear_temp_directory, create_temp_directory, get_temp_file_path, move_temp_file, resolve_temp_frame_paths
from facefusion.types import Args, ErrorCode
from facefusion.vision import pack_resolution, read_image, read_static_images, read_video_frame, restrict_image_resolution, restrict_trim_frame, restrict_video_fps, restrict_video_resolution, unpack_resolution

print(state_manager.get_item('execution_thread_count'))

# ['face_swapper', 'face_enhancer', 'expression_restorer']
args = {'command': 'headless-run', 'config_path': 'facefusion.ini', 'temp_path': '/tmp', 'jobs_path': '.jobs', 'source_paths': ['../ComfyUI/input/Sharukh.jpg'], 'target_path': '../ComfyUI/input/NagArjuna_clip_1.mp4', 'output_path': '../ComfyUI/input/FaceFusionResults/NagArjuna_clip_1_20250508_135130.mp4', 'face_detector_model': 'yolo_face', 'face_detector_size': '640x640', 'face_detector_angles': [0], 'face_detector_score': 0.5, 'face_landmarker_model': '2dfan4', 'face_landmarker_score': 0.5, 'face_selector_mode': 'reference', 'face_selector_order': 'large-small', 'face_selector_age_start': None, 'face_selector_age_end': None, 'face_selector_gender': None, 'face_selector_race': None, 'reference_face_position': 0, 'reference_face_distance': 0.3, 'reference_frame_number': 0, 'face_occluder_model': 'xseg_1', 'face_parser_model': 'bisenet_resnet_34', 'face_mask_types': ['box'], 'face_mask_blur': 0.3, 'face_mask_padding': [0, 0, 0, 0], 'face_mask_regions': ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip'], 'trim_frame_start': None, 'trim_frame_end': None, 'temp_frame_format': 'png', 'keep_temp': None, 'output_image_quality': 80, 'output_image_resolution': None, 'output_audio_encoder': 'flac', 'output_audio_quality': 80, 'output_audio_volume': 100, 'output_video_encoder': 'libx264', 'output_video_preset': 'veryfast', 'output_video_quality': 80, 'output_video_resolution': '1920x1080', 'output_video_fps': 24.0, 'processors': ['face_swapper'], 'age_modifier_model': 'styleganex_age', 'age_modifier_direction': 0, 'deep_swapper_model': 'iperov/elon_musk_224', 'deep_swapper_morph': 100, 'expression_restorer_model': 'live_portrait', 'expression_restorer_factor': 80, 'face_debugger_items': ['face-landmark-5/68', 'face-mask'], 'face_editor_model': 'live_portrait', 'face_editor_eyebrow_direction': 0.0, 'face_editor_eye_gaze_horizontal': 0.0, 'face_editor_eye_gaze_vertical': 0.0, 'face_editor_eye_open_ratio': 0.0, 'face_editor_lip_open_ratio': 0.0, 'face_editor_mouth_grim': 0.0, 'face_editor_mouth_pout': 0.0, 'face_editor_mouth_purse': 0.0, 'face_editor_mouth_smile': 0.0, 'face_editor_mouth_position_horizontal': 0.0, 'face_editor_mouth_position_vertical': 0.0, 'face_editor_head_pitch': 0.0, 'face_editor_head_yaw': 0.0, 'face_editor_head_roll': 0.0, 'face_enhancer_model': 'gfpgan_1.4', 'face_enhancer_blend': 80, 'face_enhancer_weight': 1.0, 'face_swapper_model': 'inswapper_128', 'face_swapper_pixel_boost': '1024x1024', 'frame_colorizer_model': 'ddcolor', 'frame_colorizer_size': '256x256', 'frame_colorizer_blend': 100, 'frame_enhancer_model': 'span_kendata_x4', 'frame_enhancer_blend': 80, 'lip_syncer_model': 'wav2lip_gan_96', 'execution_device_id': '0', 'execution_providers': ['cuda'], 'execution_thread_count': 8, 'execution_queue_count': 2, 'download_providers': ['github', 'huggingface'], 'video_memory_strategy': 'strict', 'system_memory_limit': 0, 'log_level': 'info'}

character_frames = [
    [140, 221],
    [270, 375],
    [688, 823],
    [844, 995],
    [1015, 1072],
    [1102, 1126],
    [1156, 1194],
    [1361, 1374],
    [1433, 1448],
    [1478, 1491]
]

def expand_ranges(ranges):
    frames = set()
    for start, end in ranges:
        frames.update(range(start, end + 1))  # inclusive
    return frames

character_set = expand_ranges(character_frames)

def frames_faceswap():

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    json_path = '../ComfyUI/input/NagArjuna.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    output_dir = f'../ComfyUI/input/FaceFusionResults/{timestamp}/'
    os.makedirs(output_dir, exist_ok=True)

    all_frame_paths = []
    process_frame_paths = []

    sorted_data = dict(sorted(data.items(), key=lambda item: int(item[0])))
    
    for key in sorted_data.keys():
        original_path = os.path.join("../ComfyUI", sorted_data[key]["original_frame"])
        filename = os.path.basename(original_path)
        new_path = os.path.join(output_dir, filename)
        shutil.copy2(original_path, new_path)
        all_frame_paths.append(new_path)
        if sorted_data[key]["character_frame"]:
            process_frame_paths.append(new_path)
    
    print("len(all_frame_paths), all_frame_paths[0], all_frame_paths[-1] ->", len(all_frame_paths), all_frame_paths[0], all_frame_paths[-1])
    print("len(process_frame_paths), process_frame_paths[0], process_frame_paths[-1] ->", len(process_frame_paths), process_frame_paths[0], process_frame_paths[-1])
    print("len(character_set) ->", len(character_set))

    source_paths = ["../ComfyUI/input/Sharukh.jpg"]

    apply_args(args, state_manager.init_item)
    print("state_manager.get_item('execution_thread_count') ->",state_manager.get_item('execution_thread_count'))
    process_manager.start()
    conditional_append_reference_faces()
    
    for processor in ['face_swapper']: #, 'face_enhancer']: #, 'expression_restorer']:

        start_time_2 = time.time()
        processor_module_1 = load_processor_module("face_swapper")
        processor_module_1.process_video(source_paths, process_frame_paths)
        processor_module_1.post_process()
        print(f"Time Taken for {processor}: {time.time() - start_time_2:.3f} seconds")

    return all_frame_paths, process_frame_paths, timestamp

def create_video_from_frames(all_frame_paths, timestamp):
    
    # Get FPS from the original video
    cap = cv2.VideoCapture('../ComfyUI/input/NagArjuna.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Get frame size from a sample frame
    sample_frame_path = all_frame_paths[0]
    sample_frame = cv2.imread(sample_frame_path)
    if sample_frame is None:
        raise ValueError(f"Sample frame not found: {sample_frame_path}")
    height, width, _ = sample_frame.shape

    output_video_path = f'../ComfyUI/input/FaceFusionResults/{timestamp}.mp4'

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write frames to video
    for i in range(len(all_frame_paths)):
        frame_path = all_frame_paths[i]
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
        else:
            print(f"Warning: Missing frame {k}, skipping.")

    out.release()
    print(f"Video saved to {output_video_path}")

    # merge_video(state_manager.get_item('target_path'), temp_video_fps, state_manager.get_item('output_video_resolution'), state_manager.get_item('output_video_fps'), trim_frame_start, trim_frame_end)

start_time = time.time()
all_frame_paths, process_frame_paths, timestamp = frames_faceswap()
print(f"Time Taken for face swapping all frames: {time.time() - start_time:.3f} seconds")
create_video_from_frames(all_frame_paths, timestamp)

# trim_frame_start, trim_frame_end = restrict_trim_frame('../ComfyUI/input/NagArjuna.mp4', state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))

# merge_video(state_manager.get_item('target_path'), temp_video_fps, state_manager.get_item('output_video_resolution'), 24, trim_frame_start, trim_frame_end)

print(f"Time Taken for full process: {time.time() - start_time:.3f} seconds")