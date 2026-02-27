import os
import random
import re
import struct
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args, monitor=False, log_dir="logs"):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.monitor = monitor
        self.log_dir = args.get("log_dir", log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.monitor else None
        self.history = dict(win=0, draw=0, lose=0, average_depth=[], max_depth=[])
        self.timing_profile_path = os.path.join(self.log_dir, "timing_profile.csv")

    def _export_onnx(self, onnx_path):
        self.model.eval()
        original_device = self.model.device
        self.model.to("cpu")
        dummy = torch.zeros(
            (1, self.game.input_channels, self.game.row_count, self.game.column_count),
            dtype=torch.float32,
        )
        torch.onnx.export(
            self.model,
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
        self.model.to(original_device)

    def _run_cpp_selfplay(self, onnx_path, memory_path, stats_path, iteration):
        cpp_bin = self.args.get("cpp_selfplay_path", "./build/cpp_selfplay")
        threads = int(self.args.get("cpp_threads", 0))
        nn_max_batch_size = int(self.args.get("cpp_nn_max_batch_size", 64))
        use_cuda = bool(self.args.get("cpp_use_cuda", torch.cuda.is_available()))
        cuda_device_id = int(self.args.get("cpp_cuda_device_id", 0))
        temp = float(self.args.get("temperature", self.args.get("chosenMoveTemperature", 1.0)))
        temp_early = float(
            self.args.get("temperature_early", self.args.get("chosenMoveTemperatureEarly", temp))
        )
        temp_halflife = float(
            self.args.get(
                "temperature_halflife", self.args.get("chosenMoveTemperatureHalflife", 19.0)
            )
        )
        seed_base = self.args.get("seed", 0)
        seed = int(seed_base + iteration)
        pcr_full_search_prob = int(self.args.get("pcr_full_search_prob", 25))
        max_game_moves = int(self.args.get("max_game_moves", 0))

        use_target_pruning = bool(self.args.get("use_target_pruning", True))
        use_fpu = bool(self.args.get("use_fpu", True))
        use_dynamic_cpuct = bool(self.args.get("use_dynamic_cpuct", True))

        fpu_reduction = float(self.args.get("fpu_reduction", 0.2))
        c_base = float(self.args.get("c_base", 19652.0))
        target_pruning_threshold = float(self.args.get("target_pruning_threshold", 0.05))

        cmd = [
            cpp_bin,
            "--onnx",
            onnx_path,
            "--out",
            memory_path,
            "--stats-out",
            stats_path,
            "--games",
            str(self.args["num_selfPlay_iterations"]),
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
            str(threads),
            "--nn-max-batch-size",
            str(nn_max_batch_size),
            "--seed",
            str(seed),
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
            cmd.extend(["--use-cuda", "--cuda-device-id", str(cuda_device_id)])
        if not use_target_pruning:
            cmd.append("--no-target-pruning")
        if not use_fpu:
            cmd.append("--no-fpu")
        if not use_dynamic_cpuct:
            cmd.append("--no-dynamic-cpuct")
        
        start_t = time.time()
        subprocess.run(cmd, check=True)
        end_t = time.time()
        elapsed = end_t - start_t
        print(f"[learn] cpp selfplay iteration={iteration} elapsed={elapsed:.2f}s")
        return elapsed

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
            path = f"./tmp_cpp_selfplay/memory_{i}.bin"
            if not os.path.exists(path):
                if i == iteration:
                    raise RuntimeError(f"Missing current memory file: {path}")
                continue
            chunk = self._load_memory_bin(path)
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

        pattern = re.compile(r"^memory_(\d+)\.bin$")
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

        stats_pattern = re.compile(r"^stats_(\d+)\.bin$")
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
        random.shuffle(memory)
        num_batches = max(1, len(memory) // self.args["batch_size"])

        for batch_idx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batch_idx : min(len(memory), batch_idx + self.args["batch_size"])]
            if len(sample) == 0:
                continue

            state, policy_targets, value_targets = zip(*sample)
            state = np.array(state, dtype=np.float32)
            policy_targets = np.array(policy_targets, dtype=np.float32)
            value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

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

    def learn(self):
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
            memory_path = f"./tmp_cpp_selfplay/memory_{iteration}.bin"
            stats_path = f"./tmp_cpp_selfplay/stats_{iteration}.bin"

            t0 = time.perf_counter()
            self._export_onnx(onnx_path)
            onnx_export_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            elapsed = self._run_cpp_selfplay(onnx_path, memory_path, stats_path, iteration)
            selfplay_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            current_memory = self._load_memory_bin(memory_path)
            memory, loaded_replay = self._load_replay_memory(iteration, current_memory=current_memory)
            memory_load_sec = time.perf_counter() - t0
            if len(current_memory) == 0:
                raise RuntimeError("C++ selfplay returned empty memory")
            if len(memory) == 0:
                raise RuntimeError("Replay memory is empty")
            removed_stale = self._cleanup_stale_memory_bins(iteration)

            t0 = time.perf_counter()
            stats = self._load_stats_bin(stats_path)
            stats_load_sec = time.perf_counter() - t0
            self.add_history(stats)
            removed_stale_stats = []
            removed_stale_onnx = []

            monitor_log_sec = 0.0
            if self.monitor:
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
            removed_stale_stats, removed_stale_onnx = self._cleanup_stale_runtime_artifacts(iteration)

            self.model.train()
            epoch_times = []
            train_start = time.perf_counter()
            for epoch in trange(self.args["num_epochs"], desc=f"train iter {iteration}"):
                epoch_start = time.perf_counter()
                self.train(memory, iteration, epoch)
                epoch_times.append(time.perf_counter() - epoch_start)
            train_total_sec = time.perf_counter() - train_start

            save_start = time.perf_counter()
            torch.save(self.model.state_dict(), f"./saved_model/model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"./saved_model/optimizer_{iteration}_{self.game}.pt")
            save_sec = time.perf_counter() - save_start

            iteration_total_sec = time.perf_counter() - iteration_start
            train_epoch_avg_sec = float(np.mean(epoch_times)) if epoch_times else 0.0
            train_epoch_min_sec = float(np.min(epoch_times)) if epoch_times else 0.0
            train_epoch_max_sec = float(np.max(epoch_times)) if epoch_times else 0.0
            rows_per_sec = (len(current_memory) / selfplay_sec) if selfplay_sec > 1e-9 else 0.0

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
            if loaded_replay:
                replay_desc = ",".join(f"{idx}:{cnt}" for idx, cnt in loaded_replay)
                print(f"[profile][iter {iteration}] replay_memory_iters={self.args.get('replay_memory_iters', 0)} loaded={replay_desc}")
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
