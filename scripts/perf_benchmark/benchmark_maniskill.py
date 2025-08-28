from typing import Dict
import numpy as np
import torch
import os
from PIL import Image

import gymnasium as gym
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

from batch_benchmark import BenchmarkArgs, write_benchmark_result_file
from benchmark_profiler import BenchmarkProfiler, get_utilization_percentages, print_system_utilization


# Get asset directory from environment variable
asset_dir = os.path.abspath(os.getenv("ASSET_DIR"))
benchmark_dir = os.path.abspath(os.path.dirname(__file__))


@register_env("SingleRobotBenchmark-v1")
class SingleRobotBenchmarkEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    SUPPORTED_ROBOTS = ["panda", "unitree_go2", "unitree_g1"]

    def __init__(
        self,
        *args,
        robot_uid="panda",
        camera_mode="minimal",
        camera_width=128,
        camera_height=128,
        camera_fov=0.7854,  # math.radian(45)
        camera_pos=(1.5, 0.5, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        **kwargs,
    ):
        self.camera_mode = camera_mode
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_pos = camera_pos
        self.camera_lookat = camera_lookat
        self.camera_fov = camera_fov
        super().__init__(*args, robot_uids=robot_uid, **kwargs)

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                uid="render_camera",
                pose=sapien_utils.look_at(self.camera_pos, self.camera_lookat),
                width=self.camera_width,
                height=self.camera_height,
                fov=self.camera_fov,
                shader_pack=self.camera_mode,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return dict()

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))

    def _load_scene(self, options: dict):
        ground_path = os.path.join(benchmark_dir, "benchmark_assets/plane_urdf/plane.urdf")
        urdf_loader = self.scene.create_urdf_loader()
        urdf_loader.fix_root_link = True
        ground_actor = urdf_loader.parse(ground_path)["actor_builders"][0]
        ground_actor._auto_inertial = True  # Force ground urdf to be static!
        self.ground = ground_actor.build_static(name="ground")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            qpos = np.zeros(len(self.agent.robot.get_active_joints()))
            self.agent.robot.set_qpos(qpos)

    def _load_lighting(self, options: Dict):
        self.scene.add_directional_light(   # norm([1.0, 1.0, -2.0] - [0, 0, 1.5])
            [0.26490647, 0.26490647, -0.92717265],  
            [3.0, 3.0, 3.0],
            shadow=False,
        )

    def _get_obs_extra(self, info: Dict):
        return dict()

    def evaluate(self):
        return {}


def main():
    args = BenchmarkArgs.parse_benchmark_args()

    env_id = "SingleRobotBenchmark-v1"
    n_envs = args.n_envs
    n_steps = args.n_steps
    sim_config = SimConfig(
        gpu_memory_config=GPUMemoryConfig(
            max_rigid_contact_count=n_envs * max(1024, n_envs) * 80,
            max_rigid_patch_count=n_envs * max(1024, n_envs) * 4,
            found_lost_pairs_capacity=2**26,
        ),
        sim_freq=100,  # dt = 0.01
        control_freq=100,  # substep = 1
        spacing=10.0,
    )

    if args.mjcf.endswith("panda.xml"):
        robot_uid = "panda"
    elif args.mjcf.endswith("go2.xml"):
        robot_uid = "unitree_go2"
    elif args.mjcf.endswith("g1.xml"):
        robot_uid = "unitree_g1"
    else:
        raise Exception(f"Invalid robot: {args.mjcf}")

    if args.rasterizer:
        camera_mode = "minimal"
    else:
        camera_mode = "rt-fast"
    obs_mode = "rgbd"  # Or "rgb"
    # render_mode = "sensors"
    render_mode = "rgb_array"  # "human for GUI"
    env = gym.make(
        env_id,
        num_envs=n_envs,
        obs_mode=obs_mode,
        render_mode=render_mode,
        control_mode="pd_joint_delta_pos",
        sim_config=sim_config,
        robot_uid=robot_uid,
        camera_mode=camera_mode,
        camera_width=args.resX,
        camera_height=args.resY,
        # parallel_in_single_scene=True,  # This actually combines all environments into a single env, is it batch rendering?
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    base_env: BaseEnv = env.unwrapped
    base_env.print_sim_details()

    # Build image output directory similar to _omni
    output_dir = os.path.dirname(args.benchmark_result_file)
    os.makedirs(output_dir, exist_ok=True)
    image_dirname = f"{args.renderer}-{args.rasterizer}-{args.n_envs}-{args.resX}"
    image_dir = os.path.join(output_dir, image_dirname)
    image_tiles = []

    profiler = BenchmarkProfiler(n_steps, n_envs)
    with torch.inference_mode():
        env.reset(seed=2022)
        for i in range(3):
            env.step(None)  # warmup step
            env.render()
        env.reset(seed=2022)
        for i in range(n_steps):
            # obs, rew, terminated, truncated, info = env.step(actions)
            print(f"Step: {i}")
            profiler.on_simulation_start()
            profiler.on_rendering_start()
            obs = env.step(None)[0]
            # When render_mode="sensors", speed of env.render() suffers a great decline in large batch size.
            profiler.on_rendering_end()
            if render_mode == "human":
                viewer = env.render()
                image_tile = None
            else:
                image_tile = env.render()
            image_tile = obs["sensor_data"]["render_camera"]["rgb"]
            
            if n_steps < 10:
                image_tiles.append(image_tile)

            if i % 10 == 0:
                system_analysis = get_utilization_percentages()
                print_system_utilization(system_analysis)

    profiler.end()
    profiler.print_summary()
    if n_steps < 10:
        os.makedirs(image_dir, exist_ok=True)
        image_tiles = [image_tile.cpu().numpy() for image_tile in image_tiles]
        for i in range(n_steps):
            for j in range(n_envs):
                image_pil = Image.fromarray(image_tiles[i][j])
                image_path = os.path.join(image_dir, f"step{i:02d}_env{j:02d}_{camera_mode}_{robot_uid}.png")
                print(f"Image saved: {image_path}")
                image_pil.save(image_path)

    env.close()
    performance_results = {
        "time_taken_gpu": profiler.get_total_rendering_gpu_time(),
        "time_taken_cpu": profiler.get_total_rendering_cpu_time(),
        "time_taken_per_env_gpu": profiler.get_total_rendering_gpu_time_per_env(),
        "time_taken_per_env_cpu": profiler.get_total_rendering_cpu_time_per_env(),
        "fps": profiler.get_rendering_fps(),
        "fps_per_env": profiler.get_rendering_fps_per_env(),
    }
    write_benchmark_result_file(args, performance_results)


if __name__ == "__main__":
    main()
