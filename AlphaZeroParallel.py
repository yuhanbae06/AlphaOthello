import os
import random
import re
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args, monitor=False, log_dir="logs"):
        self.model = model
        self.raw_model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.monitor = monitor
        self.log_dir = args.get("log_dir", log_dir)
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.ddp_enabled = False
        self.device = getattr(model, "device", torch.device("cpu"))
        self.ddp_train_only = bool(self.args.get("ddp_train_only", False))
        self.history = dict(win=0, draw=0, lose=0, average_depth=[], max_depth=[])
        self.timing_profile_path = os.path.join(self.log_dir, "timing_profile.csv")
        if self.ddp_train_only:
            self._init_distributed_if_needed()
        else:
            self.raw_model.device = self.device
        self.monitor = bool(monitor and self.is_rank0)
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.monitor else None

    @property
    def is_rank0(self):
        return self.rank == 0

    def _init_distributed_if_needed(self):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 1:
            self.raw_model.device = self.device
            return
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA")
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable in this environment")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        self.ddp_enabled = True
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)
        self.raw_model.to(self.device)
        self.raw_model.device = self.device
        self._move_optimizer_state(self.device)
        self.model = DDP(self.raw_model, device_ids=[self.local_rank], output_device=self.local_rank)

    def _move_optimizer_state(self, device):
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _barrier(self):
        if self.ddp_enabled:
            dist.barrier(device_ids=[self.local_rank])

    def _resolve_train_ddp_cuda_device_ids(self):
        config = self.args.get("train_ddp_cuda_device_ids", self.args.get("cpp_cuda_device_ids", None))
        if config is None or config == []:
            world_size = max(1, int(self.args.get("train_ddp_world_size", 1)))
            return [str(i) for i in range(world_size)]
        if isinstance(config, (list, tuple)):
            return [str(v) for v in config]
        return [str(config)]

    def _train_ddp_world_size(self):
        configured = int(self.args.get("train_ddp_world_size", 1))
        if configured > 1:
            return configured
        return len(self._resolve_train_ddp_cuda_device_ids())

    def _run_ddp_train_subprocess(self, iteration, model_path, optimizer_path):
        world_size = self._train_ddp_world_size()
        if world_size <= 1:
            return 0.0

        config_name = self.args.get("config_name")
        if not config_name:
            raise RuntimeError("config_name is required for DDP train subprocess")

        main_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Main.py")
        visible_devices = self._resolve_train_ddp_cuda_device_ids()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)

        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={len(visible_devices)}",
            main_py,
            "train-ddp",
            "--config",
            str(config_name),
            "--iteration",
            str(iteration),
            "--model-path",
            model_path,
            "--optimizer-path",
            optimizer_path,
        ]

        start_t = time.perf_counter()
        subprocess.run(cmd, check=True, env=env)
        return time.perf_counter() - start_t

    def _export_onnx(self, onnx_path):
        self.raw_model.eval()
        original_device = self.device
        self.raw_model.to("cpu")
        self.raw_model.device = torch.device("cpu")
        dummy = torch.zeros(
            (1, self.game.input_channels, self.game.row_count, self.game.column_count),
            dtype=torch.float32,
        )
        torch.onnx.export(
            self.raw_model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["policy", "value"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={
                "input": {0: "batch"},
                "policy": {0: "batch"},
                "value": {0: "batch"},
            },
            dynamo=False,
        )
        self.raw_model.to(original_device)
        self.raw_model.device = original_device

    def _normalize_int_list(self, value, key):
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        return [int(value)]

    def _resolve_cpp_threads_per_worker(self, worker_count):
        config = self.args.get("cpp_threads_per_worker", self.args.get("cpp_threads", 0))
        if isinstance(config, (list, tuple)):
            values = [int(v) for v in config]
            if len(values) != worker_count:
                raise ValueError(
                    f"cpp_threads_per_worker length ({len(values)}) must equal "
                    f"cpp_selfplay_workers ({worker_count})"
                )
            return values
        return [int(config)] * worker_count

    def _resolve_cpp_cuda_device_ids(self, use_cuda):
        if not use_cuda:
            return []
        config = self.args.get("cpp_cuda_device_ids", None)
        if config is None or config == []:
            return [int(self.args.get("cpp_cuda_device_id", 0))]
        ids = self._normalize_int_list(config, "cpp_cuda_device_ids")
        # Preserve order while dropping accidental duplicates.
        return list(dict.fromkeys(ids))

    def _validate_worker_config(self, worker_count, use_cuda):
        threads = self._resolve_cpp_threads_per_worker(worker_count)
        gpu_ids = self._resolve_cpp_cuda_device_ids(use_cuda)
        if use_cuda and len(gpu_ids) == 0:
            raise ValueError("cpp_use_cuda=true requires at least one CUDA device id")
        if use_cuda and worker_count > len(gpu_ids):
            raise ValueError(
                f"cpp_selfplay_workers ({worker_count}) cannot exceed "
                f"configured CUDA devices ({len(gpu_ids)})"
            )
        return threads, gpu_ids

    def _distribute_games_per_worker(self, total_games, worker_count):
        if worker_count <= 0:
            return []
        base = total_games // worker_count
        remainder = total_games % worker_count
        counts = []
        for worker_id in range(worker_count):
            count = base + (1 if worker_id < remainder else 0)
            counts.append(count)
        return counts

    def _worker_artifact_paths(self, iteration, worker_id):
        artifact_dir = "./tmp_cpp_selfplay"
        memory_path = os.path.join(artifact_dir, f"memory_{iteration}_w{worker_id}.bin")
        stats_path = os.path.join(artifact_dir, f"stats_{iteration}_w{worker_id}.bin")
        return memory_path, stats_path

    def _build_cpp_selfplay_cmd(self, onnx_path, worker_spec):
        cpp_bin = self.args.get("cpp_selfplay_path", "./build/cpp_selfplay")
        nn_max_batch_size = int(self.args.get("cpp_nn_max_batch_size", 64))
        use_cuda = bool(self.args.get("cpp_use_cuda", torch.cuda.is_available()))
        temp = float(self.args.get("temperature", self.args.get("chosenMoveTemperature", 1.0)))
        temp_early = float(
            self.args.get("temperature_early", self.args.get("chosenMoveTemperatureEarly", temp))
        )
        temp_halflife = float(
            self.args.get(
                "temperature_halflife", self.args.get("chosenMoveTemperatureHalflife", 19.0)
            )
        )
        pcr_full_search_prob = int(self.args.get("pcr_full_search_prob", 25))
        max_game_moves = int(self.args.get("max_game_moves", 0))

        use_target_pruning = bool(self.args.get("use_target_pruning", True))
        use_fpu = bool(self.args.get("use_fpu", True))
        use_dynamic_cpuct = bool(self.args.get("use_dynamic_cpuct", True))
        use_shaped_dirichlet = bool(self.args.get("use_shaped_dirichlet", True))

        fpu_reduction = float(self.args.get("fpu_reduction", 0.2))
        c_base = float(self.args.get("c_base", 19652.0))
        target_pruning_threshold = float(self.args.get("target_pruning_threshold", 0.05))

        cmd = [
            cpp_bin,
            "--onnx",
            onnx_path,
            "--out",
            worker_spec["memory_path"],
            "--stats-out",
            worker_spec["stats_path"],
            "--games",
            str(worker_spec["games"]),
            "--parallel-games",
            str(self.args.get("num_parallel_games", 1)),
            "--pcr-full-search-prob",
            str(pcr_full_search_prob),
            "--max-game-moves",
            str(max_game_moves),
            "--searches",
            str(self.args["num_searches"]),
            "--cpuct",
            str(self.args["C"]),
            "--temp",
            str(temp),
            "--temp-early",
            str(temp_early),
            "--temp-halflife",
            str(temp_halflife),
            "--threads",
            str(worker_spec["threads"]),
            "--nn-max-batch-size",
            str(nn_max_batch_size),
            "--seed",
            str(worker_spec["seed"]),
            "--dirichlet-epsilon",
            str(self.args["dirichlet_epsilon"]),
            "--dirichlet-alpha",
            str(self.args["dirichlet_alpha"]),
            "--target-pruning-threshold",
            str(target_pruning_threshold),
            "--fpu-reduction",
            str(fpu_reduction),
            "--c-base",
            str(c_base),
        ]
        if use_cuda:
            cmd.extend(["--use-cuda", "--cuda-device-id", str(worker_spec["gpu_id"])])
        if not use_target_pruning:
            cmd.append("--no-target-pruning")
        if not use_fpu:
            cmd.append("--no-fpu")
        if not use_dynamic_cpuct:
            cmd.append("--no-dynamic-cpuct")
        if not use_shaped_dirichlet:
            cmd.append("--no-shaped-dirichlet")
        return cmd

    def _build_selfplay_worker_specs(self, iteration, onnx_path):
        total_games = int(self.args["num_selfPlay_iterations"])
        requested_workers = max(1, int(self.args.get("cpp_selfplay_workers", 1)))
        use_cuda = bool(self.args.get("cpp_use_cuda", torch.cuda.is_available()))
        threads_per_worker, gpu_ids = self._validate_worker_config(requested_workers, use_cuda)
        games_per_worker = self._distribute_games_per_worker(total_games, requested_workers)
        seed_base = int(self.args.get("seed", 0))

        specs = []
        for worker_id, games in enumerate(games_per_worker):
            if games <= 0:
                continue
            memory_path, stats_path = self._worker_artifact_paths(iteration, worker_id)
            gpu_id = gpu_ids[worker_id] if use_cuda else None
            spec = {
                "worker_id": worker_id,
                "games": int(games),
                "threads": int(threads_per_worker[worker_id]),
                "gpu_id": gpu_id,
                "memory_path": memory_path,
                "stats_path": stats_path,
                "seed": int(seed_base + iteration * 1000003 + worker_id * 9176),
            }
            spec["cmd"] = self._build_cpp_selfplay_cmd(onnx_path, spec)
            specs.append(spec)
        return {
            "specs": specs,
            "use_cuda": use_cuda,
            "gpu_ids": gpu_ids,
        }

    def _terminate_running_workers(self, running):
        for entry in running:
            proc = entry["proc"]
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 5.0
        for entry in running:
            proc = entry["proc"]
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.05)
            if proc.poll() is None:
                proc.kill()
        for entry in running:
            stderr_thread = entry.get("stderr_thread")
            if stderr_thread is not None:
                stderr_thread.join(timeout=1.0)

    def _is_ignorable_cpp_selfplay_stderr(self, line):
        return (
            "pthread_setaffinity_np failed" in line
            or "Specify the number of threads explicitly so the affinity is not set." in line
        )

    def _launch_cpp_selfplay_worker(self, spec):
        proc = subprocess.Popen(
            spec["cmd"],
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stderr_lines = []

        def _drain_stderr():
            assert proc.stderr is not None
            for line in proc.stderr:
                if self._is_ignorable_cpp_selfplay_stderr(line):
                    continue
                stderr_lines.append(line)
                print(line, end="", file=sys.stderr)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()
        return {
            "proc": proc,
            "spec": spec,
            "start_time": time.time(),
            "stderr_lines": stderr_lines,
            "stderr_thread": stderr_thread,
        }

    def _run_cpp_selfplay_multi(self, onnx_path, iteration):
        config = self._build_selfplay_worker_specs(iteration, onnx_path)
        specs = list(config["specs"])
        if len(specs) == 0:
            raise RuntimeError("No selfplay workers were scheduled")

        use_cuda = config["use_cuda"]
        total_start = time.time()
        running = []
        completed = []
        for spec in specs:
            running.append(self._launch_cpp_selfplay_worker(spec))

        while running:
            finished_index = None
            while finished_index is None:
                for idx, entry in enumerate(running):
                    if entry["proc"].poll() is not None:
                        finished_index = idx
                        break
                if finished_index is None:
                    time.sleep(0.1)

            entry = running.pop(finished_index)
            spec = entry["spec"]
            return_code = entry["proc"].returncode
            elapsed = time.time() - entry["start_time"]
            entry["stderr_thread"].join(timeout=1.0)

            if return_code != 0:
                self._terminate_running_workers(running)
                stderr_text = "".join(entry["stderr_lines"]).strip()
                stderr_suffix = f" stderr={stderr_text}" if stderr_text else ""
                raise RuntimeError(
                    "cpp selfplay worker failed: "
                    f"iteration={iteration} worker={spec['worker_id']} gpu={spec['gpu_id']} "
                    f"returncode={return_code} out={spec['memory_path']} stats={spec['stats_path']} "
                    f"cmd={' '.join(spec['cmd'])}{stderr_suffix}"
                )

            completed.append(
                {
                    **spec,
                    "elapsed": elapsed,
                    "returncode": return_code,
                }
            )

        completed.sort(key=lambda item: item["worker_id"])
        total_elapsed = time.time() - total_start
        if use_cuda:
            assignment_summary = ",".join(
                f"w{item['worker_id']}->gpu{item['gpu_id']}"
                for item in completed
            )
        else:
            assignment_summary = ",".join(
                f"w{item['worker_id']}->cpu" for item in completed
            )
        elapsed_summary = ",".join(
            f"w{item['worker_id']}:{item['elapsed']:.1f}s" for item in completed
        )
        return {
            "elapsed": total_elapsed,
            "worker_results": completed,
            "assignment_summary": assignment_summary,
            "elapsed_summary": elapsed_summary,
        }

    def _append_timing_profile(self, row):
        os.makedirs(self.log_dir, exist_ok=True)
        header = [
            "iteration",
            "onnx_export_sec",
            "selfplay_sec",
            "memory_load_sec",
            "stats_load_sec",
            "monitor_log_sec",
            "train_total_sec",
            "train_epoch_avg_sec",
            "train_epoch_min_sec",
            "train_epoch_max_sec",
            "save_sec",
            "iteration_total_sec",
            "memory_rows",
            "train_rows",
            "selfplay_rows_per_sec",
        ]
        need_header = not os.path.exists(self.timing_profile_path)
        with open(self.timing_profile_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write(",".join(header) + "\n")
            values = [str(row.get(k, "")) for k in header]
            f.write(",".join(values) + "\n")

    def _load_memory_bin(self, memory_path):
        memory = []
        expected_state_bytes = self.game.encoded_state_size * 4
        expected_policy_bytes = self.game.action_size * 4

        with open(memory_path, "rb") as f:
            rows_buf = f.read(4)
            if len(rows_buf) != 4:
                raise RuntimeError("Invalid memory file: missing row header")
            num_rows = struct.unpack("<I", rows_buf)[0]

            for _ in range(num_rows):
                state_bytes = f.read(expected_state_bytes)
                policy_bytes = f.read(expected_policy_bytes)
                value_bytes = f.read(4)
                if (
                    len(state_bytes) != expected_state_bytes
                    or len(policy_bytes) != expected_policy_bytes
                    or len(value_bytes) != 4
                ):
                    raise RuntimeError("Invalid memory file: truncated row")

                state = np.frombuffer(state_bytes, dtype=np.float32).copy()
                policy = np.frombuffer(policy_bytes, dtype=np.float32).copy()
                value = struct.unpack("<f", value_bytes)[0]

                encoded_state = self.game.reshape_encoded_state(state)
                memory.append((encoded_state, policy, value))
        return memory

    def _artifact_paths_for_iteration(self, prefix, iteration):
        artifact_dir = "./tmp_cpp_selfplay"
        legacy_path = os.path.join(artifact_dir, f"{prefix}_{iteration}.bin")
        if os.path.exists(legacy_path):
            return [legacy_path]

        if not os.path.isdir(artifact_dir):
            return []

        pattern = re.compile(rf"^{prefix}_{iteration}_w(\d+)\.bin$")
        matches = []
        for name in os.listdir(artifact_dir):
            match = pattern.match(name)
            if match is None:
                continue
            matches.append((int(match.group(1)), os.path.join(artifact_dir, name)))
        matches.sort(key=lambda item: item[0])
        return [path for _, path in matches]

    def _load_memory_bins(self, memory_paths):
        merged_memory = []
        counts = []
        for path in memory_paths:
            chunk = self._load_memory_bin(path)
            merged_memory.extend(chunk)
            counts.append((path, len(chunk)))
        return merged_memory, counts

    def _merge_stats_dicts(self, stats_dicts):
        merged = dict(win=0, draw=0, lose=0, average_depth=[], max_depth=[], final_states=[])
        for stats in stats_dicts:
            merged["win"] += int(stats.get("win", 0))
            merged["draw"] += int(stats.get("draw", 0))
            merged["lose"] += int(stats.get("lose", 0))
            merged["average_depth"].extend(stats.get("average_depth", []))
            merged["max_depth"].extend(stats.get("max_depth", []))
            merged["final_states"].extend(stats.get("final_states", []))
        return merged

    def _load_stats_bins(self, stats_paths):
        stats_dicts = [self._load_stats_bin(path) for path in stats_paths]
        return self._merge_stats_dicts(stats_dicts)

    def _load_replay_memory(self, iteration, current_memory=None):
        replay_memory_iters = max(0, int(self.args.get("replay_memory_iters", 0)))
        start_iter = max(0, iteration - replay_memory_iters)
        merged_memory = []
        loaded = []

        for i in range(start_iter, iteration + 1):
            if i == iteration and current_memory is not None:
                merged_memory.extend(current_memory)
                loaded.append((i, len(current_memory)))
                continue
            paths = self._artifact_paths_for_iteration("memory", i)
            if len(paths) == 0:
                if i == iteration:
                    raise RuntimeError(f"Missing current memory files for iteration {i}")
                continue
            chunk, _ = self._load_memory_bins(paths)
            merged_memory.extend(chunk)
            loaded.append((i, len(chunk)))

        return merged_memory, loaded

    def _cleanup_stale_memory_bins(self, iteration):
        replay_memory_iters = max(0, int(self.args.get("replay_memory_iters", 0)))
        start_iter = max(0, iteration - replay_memory_iters)
        mem_dir = "./tmp_cpp_selfplay"
        removed = []

        if not os.path.isdir(mem_dir):
            return removed

        pattern = re.compile(r"^memory_(\d+)(?:_w\d+)?\.bin$")
        for name in os.listdir(mem_dir):
            match = pattern.match(name)
            if match is None:
                continue
            idx = int(match.group(1))
            if idx >= start_iter:
                continue
            path = os.path.join(mem_dir, name)
            try:
                os.remove(path)
                removed.append(idx)
            except OSError:
                # Ignore cleanup failures; training can continue with extra files.
                pass
        return sorted(removed)

    def _cleanup_stale_runtime_artifacts(self, iteration):
        artifact_dir = "./tmp_cpp_selfplay"
        removed_stats = []
        removed_onnx = []

        if not os.path.isdir(artifact_dir):
            return removed_stats, removed_onnx

        stats_pattern = re.compile(r"^stats_(\d+)(?:_w\d+)?\.bin$")
        onnx_pattern = re.compile(r"^model_(\d+)\.onnx$")

        for name in os.listdir(artifact_dir):
            match = stats_pattern.match(name)
            if match is not None:
                idx = int(match.group(1))
                if idx <= iteration:
                    path = os.path.join(artifact_dir, name)
                    try:
                        os.remove(path)
                        removed_stats.append(idx)
                    except OSError:
                        pass
                continue

            match = onnx_pattern.match(name)
            if match is not None:
                idx = int(match.group(1))
                if idx <= iteration:
                    path = os.path.join(artifact_dir, name)
                    try:
                        os.remove(path)
                        removed_onnx.append(idx)
                    except OSError:
                        pass

        return sorted(removed_stats), sorted(removed_onnx)

    def _load_stats_bin(self, stats_path):
        def read_u32(f):
            buf = f.read(4)
            if len(buf) != 4:
                raise RuntimeError("Invalid stats file: missing uint32")
            return struct.unpack("<I", buf)[0]

        with open(stats_path, "rb") as f:
            win = read_u32(f)
            draw = read_u32(f)
            lose = read_u32(f)

            def read_depth_lists():
                list_count = read_u32(f)
                out = []
                for _ in range(list_count):
                    length = read_u32(f)
                    if length == 0:
                        out.append([])
                        continue
                    raw = f.read(length * 4)
                    if len(raw) != length * 4:
                        raise RuntimeError("Invalid stats file: truncated depth list")
                    arr = np.frombuffer(raw, dtype=np.float32).tolist()
                    out.append(arr)
                return out

            average_depth_lists = read_depth_lists()
            max_depth_lists = read_depth_lists()

            final_state_count = read_u32(f)
            final_states = []
            if hasattr(self.game, "final_state_size"):
                expected_state_bytes = int(self.game.final_state_size())
            else:
                expected_state_bytes = self.game.row_count * self.game.column_count
            for _ in range(final_state_count):
                raw = f.read(expected_state_bytes)
                if len(raw) != expected_state_bytes:
                    raise RuntimeError("Invalid stats file: truncated final state")
                if hasattr(self.game, "decode_final_state"):
                    final_state = self.game.decode_final_state(raw)
                else:
                    final_state = np.frombuffer(raw, dtype=np.int8).copy().reshape(
                        self.game.row_count, self.game.column_count
                    )
                final_states.append(final_state)

        return dict(
            win=int(win),
            draw=int(draw),
            lose=int(lose),
            average_depth=average_depth_lists,
            max_depth=max_depth_lists,
            final_states=final_states,
        )

    def train(self, memory, num_iteration, num_epoch):
        indices = list(range(len(memory)))
        shuffle_seed = int(self.args.get("seed", 0)) + num_iteration * 1000003 + num_epoch * 9176
        random.Random(shuffle_seed).shuffle(indices)
        if self.ddp_enabled:
            indices = indices[self.rank :: self.world_size]
        num_batches = max(1, len(indices) // self.args["batch_size"])

        for batch_idx in range(0, len(indices), self.args["batch_size"]):
            batch_indices = indices[batch_idx : min(len(indices), batch_idx + self.args["batch_size"])]
            sample = [memory[i] for i in batch_indices]
            if len(sample) == 0:
                continue

            state, policy_targets, value_targets = zip(*sample)
            state = np.array(state, dtype=np.float32)
            policy_targets = np.array(policy_targets, dtype=np.float32)
            value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.device)

            out_policy, out_value = self.model(state)
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            if self.monitor:
                global_step = num_epoch * num_batches + (batch_idx // self.args["batch_size"])
                self.log_scalar(f"loss/{num_iteration}", loss.detach().cpu().item(), global_step)
                self.log_scalar(
                    f"policy_loss/{num_iteration}", policy_loss.detach().cpu().item(), global_step
                )
                self.log_scalar(
                    f"value_loss/{num_iteration}", value_loss.detach().cpu().item(), global_step
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_iteration_only(self, iteration, model_out_path, optimizer_out_path):
        current_memory, _ = self._load_memory_bins(self._artifact_paths_for_iteration("memory", iteration))
        memory, _ = self._load_replay_memory(iteration, current_memory=current_memory)
        if len(memory) == 0:
            raise RuntimeError("Replay memory is empty")

        self.model.train()
        epoch_iter = range(self.args["num_epochs"])
        if self.is_rank0:
            epoch_iter = trange(self.args["num_epochs"], desc=f"train iter {iteration}")
        for epoch in epoch_iter:
            self.train(memory, iteration, epoch)
        self._barrier()

        if self.is_rank0:
            torch.save(self.raw_model.state_dict(), model_out_path)
            torch.save(self.optimizer.state_dict(), optimizer_out_path)
        self._barrier()
        self.close_writer()
        if self.ddp_enabled and dist.is_initialized():
            dist.destroy_process_group()

    def learn(self):
        if self.ddp_train_only:
            raise RuntimeError("Use train_iteration_only() in ddp_train_only mode")
        if self.is_rank0:
            os.makedirs("./tmp_cpp_selfplay", exist_ok=True)
            os.makedirs("./saved_model", exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)

        total_iterations = int(self.args["num_iterations"])
        start_iteration = max(0, int(self.args.get("start_iteration", 0)))
        if start_iteration >= total_iterations:
            raise ValueError(
                f"start_iteration ({start_iteration}) must be smaller than "
                f"num_iterations ({total_iterations})"
            )

        for iteration in range(start_iteration, total_iterations):
            iteration_start = time.perf_counter()
            onnx_path = f"./tmp_cpp_selfplay/model_{iteration}.onnx"

            onnx_export_sec = 0.0
            selfplay_sec = 0.0
            elapsed = 0.0
            selfplay_result = None

            if self.is_rank0:
                t0 = time.perf_counter()
                self._export_onnx(onnx_path)
                onnx_export_sec = time.perf_counter() - t0

                t0 = time.perf_counter()
                selfplay_result = self._run_cpp_selfplay_multi(onnx_path, iteration)
                elapsed = selfplay_result["elapsed"]
                selfplay_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            current_memory, current_memory_chunks = self._load_memory_bins(
                self._artifact_paths_for_iteration("memory", iteration)
            )
            memory, loaded_replay = self._load_replay_memory(iteration, current_memory=current_memory)
            memory_load_sec = time.perf_counter() - t0
            if len(current_memory) == 0:
                raise RuntimeError("C++ selfplay returned empty memory")
            if len(memory) == 0:
                raise RuntimeError("Replay memory is empty")
            removed_stale = self._cleanup_stale_memory_bins(iteration) if self.is_rank0 else []

            stats_load_sec = 0.0
            stats = None
            if self.is_rank0:
                t0 = time.perf_counter()
                stats = self._load_stats_bins(self._artifact_paths_for_iteration("stats", iteration))
                stats_load_sec = time.perf_counter() - t0
                self.add_history(stats)
            removed_stale_stats = []
            removed_stale_onnx = []

            monitor_log_sec = 0.0
            if self.monitor and stats is not None:
                t0 = time.perf_counter()
                self.log_scalar("selfplay/time_sec", elapsed, iteration)
                self.log_scalar("selfplay/memory_rows", len(current_memory), iteration)
                self.log_scalar("selfplay/train_rows", len(memory), iteration)
                self.log_scalars(
                    "wining_rate",
                    {
                        "win": self.history["win"] / self.args["num_selfPlay_iterations"],
                        "lose": self.history["lose"] / self.args["num_selfPlay_iterations"],
                        "draw": self.history["draw"] / self.args["num_selfPlay_iterations"],
                    },
                    iteration,
                )
                self.log_list(
                    f"average_depth/{iteration}", self.calculate_average(self.history["average_depth"])
                )
                self.log_list(f"max_depth/{iteration}", self.calculate_average(self.history["max_depth"]))

                max_image_count = self.args.get("max_final_state_logs", 16)
                final_states = stats["final_states"]
                sample_count = min(max_image_count, len(final_states))
                if sample_count < len(final_states):
                    log_seed = int(self.args.get("seed", 0)) + 1000003 * iteration + 17
                    selected = random.Random(log_seed).sample(range(len(final_states)), sample_count)
                else:
                    selected = list(range(sample_count))
                for i, board_idx in enumerate(selected):
                    board = final_states[board_idx]
                    self.log_image(
                        f"final_state/{iteration}", self.game.get_visualized_state(board), i
                    )
                monitor_log_sec = time.perf_counter() - t0
            if self.is_rank0:
                removed_stale_stats, removed_stale_onnx = self._cleanup_stale_runtime_artifacts(iteration)

            epoch_times = []
            if self._train_ddp_world_size() > 1:
                tmp_model_path = f"./tmp_cpp_selfplay/ddp_model_{iteration}.pt"
                tmp_optimizer_path = f"./tmp_cpp_selfplay/ddp_optimizer_{iteration}.pt"
                torch.save(self.raw_model.state_dict(), tmp_model_path)
                torch.save(self.optimizer.state_dict(), tmp_optimizer_path)
                train_total_sec = self._run_ddp_train_subprocess(
                    iteration, tmp_model_path, tmp_optimizer_path
                )
                self.raw_model.load_state_dict(torch.load(tmp_model_path, map_location=self.device))
                self.optimizer.load_state_dict(torch.load(tmp_optimizer_path, map_location=self.device))
                self.raw_model.to(self.device)
                self.raw_model.device = self.device
                os.remove(tmp_model_path)
                os.remove(tmp_optimizer_path)
            else:
                self.model.train()
                train_start = time.perf_counter()
                epoch_iter = trange(self.args["num_epochs"], desc=f"train iter {iteration}")
                for epoch in epoch_iter:
                    epoch_start = time.perf_counter()
                    self.train(memory, iteration, epoch)
                    epoch_times.append(time.perf_counter() - epoch_start)
                train_total_sec = time.perf_counter() - train_start

            save_start = time.perf_counter()
            if self.is_rank0:
                torch.save(self.raw_model.state_dict(), f"./saved_model/model_{iteration}_{self.game}.pt")
                torch.save(self.optimizer.state_dict(), f"./saved_model/optimizer_{iteration}_{self.game}.pt")
            save_sec = time.perf_counter() - save_start

            iteration_total_sec = time.perf_counter() - iteration_start
            train_epoch_avg_sec = float(np.mean(epoch_times)) if epoch_times else 0.0
            train_epoch_min_sec = float(np.min(epoch_times)) if epoch_times else 0.0
            train_epoch_max_sec = float(np.max(epoch_times)) if epoch_times else 0.0
            rows_per_sec = (len(current_memory) / selfplay_sec) if selfplay_sec > 1e-9 else 0.0

            if self.is_rank0:
                print(
                    f"[profile][iter {iteration}] "
                    f"onnx_export={onnx_export_sec:.3f}s "
                    f"selfplay={selfplay_sec:.3f}s "
                    f"load_memory={memory_load_sec:.3f}s "
                    f"load_stats={stats_load_sec:.3f}s "
                    f"monitor_log={monitor_log_sec:.3f}s "
                    f"train={train_total_sec:.3f}s "
                    f"save={save_sec:.3f}s "
                    f"total={iteration_total_sec:.3f}s "
                    f"rows={len(current_memory)} "
                    f"train_rows={len(memory)} "
                    f"rows_per_sec={rows_per_sec:.1f}"
                )
                print(
                    f"[profile][iter {iteration}] "
                    f"selfplay_gpu_assignments={selfplay_result['assignment_summary']}"
                )
                print(
                    f"[profile][iter {iteration}] "
                    f"selfplay_worker_sec={selfplay_result['elapsed_summary']}"
                )
            if self.is_rank0:
                if current_memory_chunks:
                    chunk_desc = ",".join(
                        f"{os.path.basename(path)}:{count}" for path, count in current_memory_chunks
                    )
                    print(f"[profile][iter {iteration}] selfplay_memory_chunks={chunk_desc}")
                if loaded_replay:
                    replay_desc = ",".join(f"{idx}:{cnt}" for idx, cnt in loaded_replay)
                    print(
                        f"[profile][iter {iteration}] replay_memory_iters="
                        f"{self.args.get('replay_memory_iters', 0)} loaded={replay_desc}"
                    )
                if removed_stale:
                    print(f"[profile][iter {iteration}] removed_stale_memory_bins={removed_stale}")
                if removed_stale_stats:
                    print(f"[profile][iter {iteration}] removed_stale_stats_bins={removed_stale_stats}")
                if removed_stale_onnx:
                    print(f"[profile][iter {iteration}] removed_stale_onnx_models={removed_stale_onnx}")
                print(
                    f"[profile][iter {iteration}] "
                    f"train_epoch_sec min/avg/max="
                    f"{train_epoch_min_sec:.3f}/{train_epoch_avg_sec:.3f}/{train_epoch_max_sec:.3f}"
                )

            timing_row = dict(
                iteration=iteration,
                onnx_export_sec=f"{onnx_export_sec:.6f}",
                selfplay_sec=f"{selfplay_sec:.6f}",
                memory_load_sec=f"{memory_load_sec:.6f}",
                stats_load_sec=f"{stats_load_sec:.6f}",
                monitor_log_sec=f"{monitor_log_sec:.6f}",
                train_total_sec=f"{train_total_sec:.6f}",
                train_epoch_avg_sec=f"{train_epoch_avg_sec:.6f}",
                train_epoch_min_sec=f"{train_epoch_min_sec:.6f}",
                train_epoch_max_sec=f"{train_epoch_max_sec:.6f}",
                save_sec=f"{save_sec:.6f}",
                iteration_total_sec=f"{iteration_total_sec:.6f}",
                memory_rows=len(current_memory),
                train_rows=len(memory),
                selfplay_rows_per_sec=f"{rows_per_sec:.3f}",
            )
            if self.is_rank0:
                self._append_timing_profile(timing_row)

            if self.monitor:
                self.log_scalar("timing/onnx_export_sec", onnx_export_sec, iteration)
                self.log_scalar("timing/selfplay_sec", selfplay_sec, iteration)
                self.log_scalar("timing/memory_load_sec", memory_load_sec, iteration)
                self.log_scalar("timing/stats_load_sec", stats_load_sec, iteration)
                self.log_scalar("timing/monitor_log_sec", monitor_log_sec, iteration)
                self.log_scalar("timing/train_total_sec", train_total_sec, iteration)
                self.log_scalar("timing/train_epoch_avg_sec", train_epoch_avg_sec, iteration)
                self.log_scalar("timing/train_epoch_min_sec", train_epoch_min_sec, iteration)
                self.log_scalar("timing/train_epoch_max_sec", train_epoch_max_sec, iteration)
                self.log_scalar("timing/save_sec", save_sec, iteration)
                self.log_scalar("timing/iteration_total_sec", iteration_total_sec, iteration)
                self.log_scalar("timing/selfplay_rows_per_sec", rows_per_sec, iteration)

            if self.is_rank0:
                self.reset_history()

        self.close_writer()

    def add_history(self, return_history):
        self.history["win"] += int(return_history.get("win", 0))
        self.history["draw"] += int(return_history.get("draw", 0))
        self.history["lose"] += int(return_history.get("lose", 0))
        for depth_list in return_history.get("average_depth", []):
            self.history["average_depth"].append(depth_list)
        for depth_list in return_history.get("max_depth", []):
            self.history["max_depth"].append(depth_list)

    def reset_history(self):
        self.history = dict(win=0, draw=0, lose=0, average_depth=[], max_depth=[])

    def calculate_average(self, depth_lists):
        if len(depth_lists) == 0:
            return []
        return_list = []
        lengths = np.array([len(x) for x in depth_lists], dtype=np.int32)
        max_len = int(lengths.max()) if len(lengths) > 0 else 0
        for i in range(max_len):
            val = 0.0
            cnt = 0
            for depth_list in depth_lists:
                if len(depth_list) > i:
                    val += depth_list[i]
                    cnt += 1
            if cnt > 0:
                return_list.append(val / cnt)
        return return_list

    def log_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag, values, step):
        if self.writer is not None:
            self.writer.add_scalars(tag, values, step)

    def log_list(self, tag, value_list):
        if self.writer is None:
            return
        for i, value in enumerate(value_list):
            self.writer.add_scalar(tag, value, i)

    def log_image(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_image(tag, value, step)

    def close_writer(self):
        if self.writer is not None:
            self.writer.close()

