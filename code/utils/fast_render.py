import os
import time
import numpy as np
import pyrender
import trimesh
import queue
import imageio
import threading
import multiprocessing
import utils.media
import glob

def deg_to_rad(degrees):
    return degrees * np.pi / 180

def create_pose_camera(angle_deg):
    angle_rad = deg_to_rad(angle_deg)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle_rad), -np.sin(angle_rad), 0.2],  # 降低Y轴位置
        [0.0, np.sin(angle_rad), np.cos(angle_rad), 6.0],  # 增加Z轴距离，拉远镜头
        [0.0, 0.0, 0.0, 1.0]
    ])

def create_pose_light(angle_deg):
    angle_rad = deg_to_rad(angle_deg)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle_rad), -np.sin(angle_rad), 0.0],
        [0.0, np.sin(angle_rad), np.cos(angle_rad), 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def create_scene_with_mesh(vertices, faces, uniform_color, pose_camera, pose_light):
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=uniform_color)
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=True)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    scene.add(camera, pose=pose_camera)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    scene.add(light, pose=pose_light)
    return scene

def do_render_one_frame(renderer, frame_idx, vertices, vertices1, faces):
    if frame_idx % 100 == 0:
        print('processed', frame_idx, 'frames')

    uniform_color = [220, 220, 220, 255]
    pose_camera = create_pose_camera(angle_deg=-4)  # 增加俯视角
    pose_light = create_pose_light(angle_deg=-30)

    figs = []
    for vtx in [vertices, vertices1]:
        # print(vtx.shape)
        scene = create_scene_with_mesh(vtx, faces, uniform_color, pose_camera, pose_light)
        fig, _ = renderer.render(scene)
        figs.append(fig)
  
    return figs[0], figs[1]

def do_render_one_frame_no_gt(renderer, frame_idx, vertices, faces):
    if frame_idx % 100 == 0:
        print('processed', frame_idx, 'frames')

    uniform_color = [220, 220, 220, 255]
    pose_camera = create_pose_camera(angle_deg=-4)  # 增加俯视角
    pose_light = create_pose_light(angle_deg=-30)

    figs = []
    # for vtx in [vertices]:
    #     print(vtx.shape)
    # print(vertices.shape)
    scene = create_scene_with_mesh(vertices, faces, uniform_color, pose_camera, pose_light)
    fig, _ = renderer.render(scene)
    figs.append(fig)
  
    return figs[0]

def write_images_from_queue(fig_queue, output_dir, img_filetype):
    while True:
        e = fig_queue.get()
        if e is None:
            break
        fid, fig1, fig2 = e
        filename = os.path.join(output_dir, f"frame_{fid}.{img_filetype}")
        merged_fig = np.hstack((fig1, fig2))
        try:
            imageio.imwrite(filename, merged_fig)
        except Exception as ex:
            print(f"Error writing image {filename}: {ex}")
            raise ex
        
def write_images_from_queue_no_gt(fig_queue, output_dir, img_filetype):
    while True:
        e = fig_queue.get()
        if e is None:
            break
        fid, fig1, fig2 = e
        filename = os.path.join(output_dir, f"frame_{fid}.{img_filetype}")
        merged_fig = fig1 #np.hstack((fig1))
        try:
            imageio.imwrite(filename, merged_fig)
        except Exception as ex:
            print(f"Error writing image {filename}: {ex}")
            raise ex
        
    
def render_frames_and_enqueue(fids, frame_vertex_pairs, faces, render_width, render_height, fig_queue):
    fig_resolution = (render_width // 2, render_height)
    renderer = pyrender.OffscreenRenderer(*fig_resolution)

    for idx, fid in enumerate(fids):
        fig1, fig2 = do_render_one_frame(renderer, fid, frame_vertex_pairs[idx][0], frame_vertex_pairs[idx][1], faces)
        fig_queue.put((fid, fig1, fig2))
    
    renderer.delete()

def render_frames_and_enqueue_no_gt(fids, frame_vertex_pairs, faces, render_width, render_height, fig_queue):
    fig_resolution = (render_width // 2, render_height)
    renderer = pyrender.OffscreenRenderer(*fig_resolution)

    for idx, fid in enumerate(fids):
        fig1 = do_render_one_frame_no_gt(renderer, fid, frame_vertex_pairs[idx][0], faces)
        fig_queue.put((fid, fig1))
    
    renderer.delete()

def sub_process_process_frame(subprocess_index, render_video_width, render_video_height, render_tmp_img_filetype, fids, frame_vertex_pairs, faces, output_dir):
    begin_ts = time.time()
    print(f"subprocess_index={subprocess_index} begin_ts={begin_ts}")

    fig_queue = queue.Queue()
    render_frames_and_enqueue(fids, frame_vertex_pairs, faces, render_video_width, render_video_height, fig_queue)
    fig_queue.put(None)
    render_end_ts = time.time()

    image_writer_thread = threading.Thread(target=write_images_from_queue, args=(fig_queue, output_dir, render_tmp_img_filetype))
    image_writer_thread.start()
    image_writer_thread.join()

    write_end_ts = time.time()
    print(
        f"subprocess_index={subprocess_index} "
        f"render={render_end_ts - begin_ts:.2f} "
        f"all={write_end_ts - begin_ts:.2f} "
        f"begin_ts={begin_ts:.2f} "
        f"render_end_ts={render_end_ts:.2f} "
        f"write_end_ts={write_end_ts:.2f}"
    )

def sub_process_process_frame_no_gt(subprocess_index, render_video_width, render_video_height, render_tmp_img_filetype, fids, frame_vertex_pairs, faces, output_dir):
    begin_ts = time.time()
    print(f"subprocess_index={subprocess_index} begin_ts={begin_ts}")

    fig_queue = queue.Queue()
    render_frames_and_enqueue(fids, frame_vertex_pairs, faces, render_video_width, render_video_height, fig_queue)
    fig_queue.put(None)
    render_end_ts = time.time()

    image_writer_thread = threading.Thread(target=write_images_from_queue_no_gt, args=(fig_queue, output_dir, render_tmp_img_filetype))
    image_writer_thread.start()
    image_writer_thread.join()

    write_end_ts = time.time()
    print(
        f"subprocess_index={subprocess_index} "
        f"render={render_end_ts - begin_ts:.2f} "
        f"all={write_end_ts - begin_ts:.2f} "
        f"begin_ts={begin_ts:.2f} "
        f"render_end_ts={render_end_ts:.2f} "
        f"write_end_ts={write_end_ts:.2f}"
    )

def distribute_frames(frames, render_video_fps, render_concurent_nums, vertices_all, vertices1_all):
    sample_interval = max(1, int(30 // render_video_fps))
    subproc_frame_ids = [[] for _ in range(render_concurent_nums)]
    subproc_vertices = [[] for _ in range(render_concurent_nums)]
    sampled_frame_id = 0

    for i in range(frames):
        if i % sample_interval != 0:
            continue
        subprocess_index = sampled_frame_id % render_concurent_nums
        subproc_frame_ids[subprocess_index].append(sampled_frame_id)
        subproc_vertices[subprocess_index].append((vertices_all[i], vertices1_all[i]))
        sampled_frame_id += 1

    return subproc_frame_ids, subproc_vertices

def distribute_frames_no_gt(frames, render_video_fps, render_concurent_nums, vertices_all):
    sample_interval = max(1, int(30 // render_video_fps))
    subproc_frame_ids = [[] for _ in range(render_concurent_nums)]
    subproc_vertices = [[] for _ in range(render_concurent_nums)]
    sampled_frame_id = 0

    for i in range(frames):
        if i % sample_interval != 0:
            continue
        subprocess_index = sampled_frame_id % render_concurent_nums
        subproc_frame_ids[subprocess_index].append(sampled_frame_id)
        subproc_vertices[subprocess_index].append((vertices_all[i], vertices_all[i]))
        sampled_frame_id += 1

    return subproc_frame_ids, subproc_vertices

def generate_silent_videos(render_video_fps, 
                           render_video_width,
                           render_video_height,
                           render_concurent_nums,
                           render_tmp_img_filetype,
                           frames, 
                           vertices_all,
                           vertices1_all,
                           faces,
                           output_dir):

    subproc_frame_ids, subproc_vertices = distribute_frames(frames, render_video_fps, render_concurent_nums, vertices_all, vertices1_all)

    print(f"generate_silent_videos concurrentNum={render_concurent_nums} time={time.time()}")
    with multiprocessing.Pool(render_concurent_nums) as pool:
        pool.starmap(
            sub_process_process_frame, 
            [
                (subprocess_index,  render_video_width, render_video_height, render_tmp_img_filetype, subproc_frame_ids[subprocess_index],  subproc_vertices[subprocess_index], faces, output_dir) 
                    for subprocess_index in range(render_concurent_nums)
            ]
        )

    output_file = os.path.join(output_dir, "silence_video.mp4")
    utils.media.convert_img_to_mp4(os.path.join(output_dir, f"frame_%d.{render_tmp_img_filetype}"), output_file, render_video_fps)
    filenames = glob.glob(os.path.join(output_dir, f"*.{render_tmp_img_filetype}"))
    for filename in filenames:
        os.remove(filename)

    return output_file

def generate_silent_videos_no_gt(render_video_fps, 
                           render_video_width,
                           render_video_height,
                           render_concurent_nums,
                           render_tmp_img_filetype,
                           frames, 
                           vertices_all,
                           faces,
                           output_dir):

    subproc_frame_ids, subproc_vertices = distribute_frames_no_gt(frames, render_video_fps, render_concurent_nums, vertices_all)

    print(f"generate_silent_videos concurrentNum={render_concurent_nums} time={time.time()}")
    #sub_process_process_frame_no_gt(0,  render_video_width, render_video_height, render_tmp_img_filetype, subproc_frame_ids[0],  subproc_vertices[0], faces, output_dir) 
    with multiprocessing.Pool(render_concurent_nums) as pool:
        pool.starmap(
            sub_process_process_frame_no_gt, 
            [
                (subprocess_index,  render_video_width, render_video_height, render_tmp_img_filetype, subproc_frame_ids[subprocess_index],  subproc_vertices[subprocess_index], faces, output_dir) 
                    for subprocess_index in range(render_concurent_nums)
            ]
        )

    output_file = os.path.join(output_dir, "silence_video.mp4")
    utils.media.convert_img_to_mp4(os.path.join(output_dir, f"frame_%d.{render_tmp_img_filetype}"), output_file, render_video_fps)
    filenames = glob.glob(os.path.join(output_dir, f"*.{render_tmp_img_filetype}"))
    for filename in filenames:
        os.remove(filename)

    return output_file