#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MultimodalSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import List, Literal, Tuple, Optional
from collections import defaultdict
import os
import stat

import numpy as np
import torch
import cv2
from transformers import (
    ChineseCLIPProcessor,
    ChineseCLIPModel,
    CLIPProcessor,
    CLIPModel,
)

DECAY_STEP = 0.01
INITIAL_STAGE_DIFF = 0.03
STAGE_DECAY_STEP = 0.0015
DIVERSITY_DISTANCE_THRESHOLD = 0.3


class BaseFrameSelector(ABC):
    """Base class for keyframe selectors using CLIP-based text-image matching.

    Pipeline:
        1. Load CLIP/CN-CLIP model and extract text & image features.
        2. Compute cosine similarity between query and each frame.
        3. Detect scene boundaries via image-feature similarity gradient.
        4. Subclasses implement specific selection strategies (interval or discrete).

    Attributes:
        similar_threshold: Text-image similarity decay threshold for relevance filtering.
        image_similar_threshold: Image-image similarity gradient threshold for scene
            boundary detection. A scene change is detected when the cosine similarity
            gradient between adjacent frames exceeds this value.
    """

    def __init__(
        self,
        model_path: str,
        device_id: int,
        model_type: str = "clip",
        similar_threshold: float = 0.025,
        image_similar_threshold: float = 0.015,
        image_size: tuple = (672, 672),
    ):
        """Initializes the frame selector.

        Args:
            model_path: Path to CLIP model weights.
            device_id: NPU device index for inference.
            model_type: Model variant — "clip" (English) or "cn_clip" (Chinese).
            similar_threshold: Text-image similarity decay threshold. Frames with
                similarity below (max_similarity - this value) are filtered out.
            image_similar_threshold: Image similarity gradient threshold for scene
                boundary detection.
            image_size: Resize target for input images, must match model training size.
        """
        self.model_path = model_path
        self.device_id = device_id
        self.model_type = model_type
        self.batch_size = 64
        self.similar_threshold = similar_threshold
        self.image_similar_threshold = image_similar_threshold
        self.image_size = image_size
        self.device = f'npu:{device_id}'
        self.processor = None
        self.model = None
        self.model_initializers = {
            "clip": self._init_clip_model,
            "cn_clip": self._init_cn_clip_model,
        }

        self._check_input_valid()
        self._init_model()

    @abstractmethod
    def select_keyframes(
        self, query: str, frames: List[np.ndarray], sample_num: int, do_resample: bool
    ) -> Tuple[List[int], List[np.ndarray]]:
        """Selects keyframes relevant to the query from a video frame sequence.

        Args:
            query: User query text describing the target visual content.
            frames: List of video frames as BGR numpy arrays.
            sample_num: Maximum number of keyframes to select.
            do_resample: Whether to perform adaptive resampling within intervals.

        Returns:
            A tuple of (keyframe indices, keyframe images).
        """
        pass

    def _validate_select_inputs(self, query, frames, sample_num):
        """Validates inputs for select_keyframes.

        Args:
            query: Query text.
            frames: Video frame list.
            sample_num: Desired sample count.

        Returns:
            Capped sample count (min of sample_num and frame count).
        """
        if not query or not frames:
            raise ValueError("query and frames must not be empty")
        if not isinstance(query, str):
            raise TypeError("query must be string")
        if not isinstance(sample_num, int) or sample_num <= 0:
            raise ValueError("sample_num must be positive integer")
        return min(sample_num, len(frames))

    def _compute_similarities(self, query: str, frames: List[np.ndarray]):
        """Computes cosine similarity between the query and all frames.

        Args:
            query: Query text.
            frames: Video frame list.

        Returns:
            Tuple of (query_feature, image_features, query_similarities):
                query_feature: L2-normalized text feature (1, D).
                image_features: L2-normalized image features (N, D).
                query_similarities: Per-frame cosine similarity with query (N,).
        """
        query_feature, image_features = self._extract_features(query, frames)
        query_similarities = torch.cosine_similarity(query_feature, image_features).float()
        return query_feature, image_features, query_similarities

    def _check_input_valid(self):
        """Validates initialization parameters."""
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("model_path must be a non-empty string")
        if not isinstance(self.device_id, int):
            raise TypeError("device_id must be an integer")
        if not isinstance(self.model_type, str) or not self.model_type.strip():
            raise ValueError("model_type must be a non-empty string")
        if self.model_type not in self.model_initializers:
            raise ValueError(f"model_type must be in {self.model_initializers.keys()}")
        if not isinstance(self.similar_threshold, (int, float)):
            raise TypeError("similar_threshold must be a number")
        if not (0 <= self.similar_threshold <= 1.0):
            raise ValueError("similar_threshold must be in [0, 1]")
        if not isinstance(self.image_similar_threshold, (int, float)):
            raise TypeError("image_similar_threshold must be a number")
        if not (0 <= self.image_similar_threshold <= 1.0):
            raise ValueError("image_similar_threshold must be in [0, 1]")
        if not isinstance(self.image_size, tuple) or len(self.image_size) != 2:
            raise ValueError("image_size must be a tuple of two integers (width, height)")
        if not all(isinstance(v, int) for v in self.image_size):
            raise TypeError("image_size values must be integers")
        if not (10 <= self.image_size[0] <= 8192 and 10 <= self.image_size[1] <= 8192):
            raise ValueError("image_size width and height must be in [10, 8192]")
        self._validate_model_file_security()

    def _validate_model_file_security(self):
        """Validates model directory ownership and permissions for security.

        Checks that model_path is a directory owned by the current user with
        exactly 640 (rw- r-- ---) permissions, preventing unauthorized
        modification or access by other users.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model_path does not exist: {self.model_path}")

        if not os.path.isdir(self.model_path):
            raise ValueError(f"model_path must be a directory, got: {self.model_path}")

        current_uid = os.getuid()  # pylint: disable=all
        path_stat = os.stat(self.model_path)
        if path_stat.st_uid != current_uid:
            raise PermissionError(
                f"Model directory '{self.model_path}' owner (uid={path_stat.st_uid}) "
                f"does not match current user (uid={current_uid})"
            )
        file_mode = stat.S_IMODE(path_stat.st_mode)
        if file_mode != 0o640:
            raise PermissionError(f"Model directory '{self.model_path}' permissions must be 640, got {oct(file_mode)}")

    def _init_model(self):
        """Dispatches to the model initializer matching model_type."""
        if self.model_type not in self.model_initializers:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. Supported: {list(self.model_initializers.keys())}"
            )
        self.model_initializers[self.model_type]()

    def _init_clip_model(self):
        """Loads English CLIP model (FP16) onto NPU."""
        self.processor = CLIPProcessor.from_pretrained(self.model_path)
        self.model = CLIPModel.from_pretrained(
            self.model_path, torch_dtype=torch.float16, device_map=self.device
        ).eval()

    def _init_cn_clip_model(self):
        """Loads Chinese CLIP model (FP16) onto NPU."""
        self.processor = ChineseCLIPProcessor.from_pretrained(self.model_path)
        self.model = ChineseCLIPModel.from_pretrained(
            self.model_path, torch_dtype=torch.float16, device_map=self.device
        ).eval()

    def _extract_features(self, query: str, frames: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts L2-normalized text and image features.

        Args:
            query: Query text.
            frames: Video frame list.

        Returns:
            Tuple of (query_feature, image_features).
        """
        query_features = self._extract_query_features(query)
        image_features = self._extract_image_features(frames)
        return query_features, image_features

    def _extract_query_features(self, query: str):
        """Extracts and L2-normalizes CLIP text features for the query.

        Args:
            query: Query text string.

        Returns:
            Normalized text feature vector of shape (1, D).
        """
        clip_inputs = self.processor(text=query, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**clip_inputs)
        query_feature = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return query_feature

    def _extract_image_features(self, frames):
        """Extracts and L2-normalizes CLIP image features in batches.

        Steps:
            1. Process frames in batches of self.batch_size to avoid OOM.
            2. Resize each frame to self.image_size.
            3. Concatenate features from all batches.
            4. L2-normalize the full feature matrix.

        Args:
            frames: List of BGR numpy arrays.

        Returns:
            Normalized image feature matrix of shape (N, D).
        """
        all_features = []
        for i in range(0, len(frames), self.batch_size):
            batch_images = frames[i : i + self.batch_size]
            if self.image_size:
                batch_images = [cv2.resize(frame, self.image_size) for frame in batch_images]
            inputs_image = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs_image)
            all_features.append(image_features)

        image_features = torch.cat(all_features, dim=0)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def _find_boundary(
        self,
        image_features: torch.Tensor,
        candidate_idx: int,
        direction: Literal['left', 'right'] = 'left',
        step: int = 10,
        max_span: int = 300,
    ) -> int:
        """Searches for a scene boundary from a candidate frame in the given direction.

        Algorithm:
            Uses the candidate frame's image feature as an anchor. Scans left/right
            in chunks of `step` frames, computing cosine similarity between the anchor
            and each scanned frame. When the first-order difference (gradient) of the
            similarity sequence exceeds `image_similar_threshold`, a scene change is
            detected at that position.

        Args:
            image_features: Normalized image features for all frames (N, D).
            candidate_idx: Index of the candidate frame (search start).
            direction: 'left' or 'right'.
            step: Number of frames per scan chunk.
            max_span: Maximum number of frames to scan.

        Returns:
            Frame index of the detected boundary. Returns 0 if left boundary
            reaches the start, or (N-1) if right boundary reaches the end.
        """
        total = len(image_features)
        if direction == 'left' and candidate_idx <= 0:
            return 0
        elif direction == 'right' and candidate_idx >= total - 1:
            return total - 1
        anchor = image_features[candidate_idx].unsqueeze(0)
        current_pos = candidate_idx - 1 if direction == 'left' else candidate_idx + 1
        prev_sim = None

        frames_scanned = 0

        while frames_scanned < max_span:
            if direction == 'left':
                end = current_pos + 1
                start = max(0, current_pos - step + 1)
            else:
                start = current_pos
                end = min(total, current_pos + step)

            chunk_len = end - start
            if chunk_len <= 0:
                break

            chunk_feats = image_features[start:end]
            sims = torch.cosine_similarity(anchor, chunk_feats, dim=1)

            # Prepend previous chunk's tail similarity (or 1.0) to compute
            # the gradient at the chunk boundary.
            if prev_sim is not None:
                extended_sims = torch.cat([torch.tensor([prev_sim], device=sims.device), sims])
            else:
                extended_sims = torch.cat([torch.ones(1, device=sims.device), sims])

            # Detect scene change via similarity gradient.
            grad = torch.abs(torch.diff(extended_sims))
            exceed_mask = grad > self.image_similar_threshold

            if exceed_mask.any():
                # Left: take the last exceeding point (closest to anchor).
                # Right: take the first exceeding point (earliest boundary).
                exceed_indices = torch.where(exceed_mask)[0]
                rel_idx = exceed_indices[-1].item() + 1 if direction == 'left' else exceed_indices[0].item() + 1
                return start + (rel_idx - 1)

            prev_sim = sims[-1].item() if direction == 'left' else sims[0].item()
            current_pos += -step if direction == 'left' else step
            frames_scanned += step

        if direction == "left":
            return 0
        return total - 1

    def _find_left(self, image_features: torch.Tensor, candidate_idx: int, step: int = 10) -> int:
        """Searches leftward for a scene boundary.

        Args:
            image_features: Normalized image features (N, D).
            candidate_idx: Candidate frame index.
            step: Scan step size.

        Returns:
            Left boundary frame index.
        """
        return int(self._find_boundary(image_features, candidate_idx, 'left', step))

    def _find_right(self, image_features: torch.Tensor, candidate_idx: int, step: int = 10) -> int:
        """Searches rightward for a scene boundary.

        Args:
            image_features: Normalized image features (N, D).
            candidate_idx: Candidate frame index.
            step: Scan step size.

        Returns:
            Right boundary frame index.
        """
        return int(self._find_boundary(image_features, candidate_idx, 'right', step))


class KRangFrameSelector(BaseFrameSelector):
    """Interval-based keyframe selector: locates continuous scene intervals relevant
    to the query and adaptively samples keyframes within each interval.

    Unlike KFrameSelector which outputs discrete frames, this selector outputs
    continuous intervals (scene segments) suitable for tasks requiring temporal context.

    Pipeline:
        1. Compute per-frame query similarity.
        2. Greedily select diverse high-similarity candidates and expand to intervals.
        3. Merge adjacent semantically similar intervals.
        4. Adaptively resample within merged intervals.
    """

    @staticmethod
    def _get_sample_num(diff):
        """Returns base sample count based on interval length.

        Args:
            diff: Interval length in frames.

        Returns:
            12 for long intervals (>=30 frames), 6 otherwise.
        """
        if diff >= 30:
            return 12
        return 6

    @staticmethod
    def _adaptive_resample(
        interval: List[int],
        image_features: torch.Tensor,
        query_feature: torch.Tensor,
        base_sample_num: int = 1,
        top_ratio: float = 0.4,
    ) -> List[int]:
        """Adaptively resamples within an interval, balancing relevance and coverage.

        Three-phase strategy:
            Phase 1 - Relevance: If interval exceeds quota, select top-k by similarity.
            Phase 2 - Top-ratio supplement: Fill top_ratio of extra quota with
                highest-similarity frames.
            Phase 3 - Uniform supplement: Fill remaining quota with uniformly
                spaced frames for temporal coverage.

        Args:
            interval: Frame indices within the interval.
            image_features: Normalized image features for all frames.
            query_feature: Normalized query text feature.
            base_sample_num: Sampling quota for this interval.
            top_ratio: Fraction of extra quota allocated to high-similarity frames.

        Returns:
            Sorted list of resampled frame indices.
        """
        if not interval:
            return []
        interval_set = set(interval)
        original_count = len(interval_set)

        # Phase 1: interval exceeds quota — select top-k by similarity
        if original_count > base_sample_num:
            min_idx, max_idx = min(interval), max(interval)
            interval_features = image_features[min_idx : max_idx + 1]
            similarities = torch.cosine_similarity(query_feature, interval_features)
            _, top_indices = torch.topk(similarities, base_sample_num)
            selected = [min_idx + idx.item() for idx in top_indices]
            return sorted(selected)

        extra_quota = base_sample_num - original_count

        if extra_quota <= 0:
            return sorted(interval_set)

        min_idx, max_idx = min(interval), max(interval)
        interval_length = max_idx - min_idx + 1
        interval_features = image_features[min_idx : max_idx + 1]
        similarities = torch.cosine_similarity(query_feature, interval_features)

        # Frames not yet selected (available for supplemental sampling)
        available_rel_indices = [i for i in range(interval_length) if min_idx + i not in interval_set]

        if not available_rel_indices:
            return sorted(interval_set)

        # Phase 2: select top_count high-similarity frames from available pool
        top_count = max(1, int(extra_quota * top_ratio))
        top_count = min(top_count, len(available_rel_indices))

        available_sim = similarities[available_rel_indices]
        _, top_indices = torch.topk(available_sim, top_count)
        selected_rel = [available_rel_indices[i.item()] for i in top_indices]

        # Phase 3: uniformly sample remaining quota for temporal coverage
        remaining = extra_quota - len(selected_rel)
        if remaining > 0 and len(available_rel_indices) > len(selected_rel):
            if remaining >= len(available_rel_indices):
                remaining_pool = [i for i in available_rel_indices if i not in selected_rel]
                selected_rel.extend(remaining_pool)
            else:
                unselected_pool = [i for i in available_rel_indices if i not in selected_rel]
                if len(unselected_pool) > 1:
                    uniform_indices = np.linspace(0, len(unselected_pool) - 1, remaining, dtype=np.int32)
                    selected_rel.extend(unselected_pool[i] for i in uniform_indices)
                elif unselected_pool:
                    selected_rel.append(unselected_pool[0])

        # Merge original and sampled frames, truncate to quota
        final_set = interval_set.copy()
        for rel_idx in selected_rel:
            final_set.add(min_idx + rel_idx)

        final_indices = sorted(final_set)
        if len(final_indices) > base_sample_num:
            return final_indices[:base_sample_num]

        return final_indices

    def select_keyframes(
        self, query: str, frames: List[np.ndarray], sample_num: int, do_resample: bool = True
    ) -> Tuple[List[int], List[np.ndarray]]:
        """Interval-based keyframe selection pipeline.

        Args:
            query: Query text.
            frames: Video frame list.
            sample_num: Maximum keyframe count.
            do_resample: Whether to adaptively resample within intervals.

        Returns:
            Tuple of (keyframe indices, keyframe images).
        """
        sample_num = self._validate_select_inputs(query, frames, sample_num)

        # Step 1: extract features and compute per-frame similarity
        query_feature, image_features, query_similarities = self._compute_similarities(query, frames)

        # Step 2: greedily select diverse candidates and expand to scene intervals
        frame_indices = self._select_diverse_frames(query_similarities, image_features, sample_num)

        # Step 3: merge adjacent semantically similar intervals
        merged_frame_indices = self._merge_intervals(frame_indices, query_similarities=query_similarities)

        # Step 4: adaptively resample within merged intervals
        if do_resample:
            resampled_frame_indices = self._resample(merged_frame_indices, image_features, query_feature, sample_num)
        else:
            resampled_frame_indices = merged_frame_indices
        unique_indices = sorted(set(resampled_frame_indices))
        key_frames = [frames[i] for i in unique_indices]
        return unique_indices, key_frames

    def _select_diverse_frames(
        self,
        query_similarities: torch.Tensor,
        image_features: torch.Tensor,
        sample_num: int,
        use_threshold: bool = True,
    ) -> List[List[int]]:
        """Greedily selects diverse high-similarity candidates and expands each to a scene interval.

        Algorithm:
            1. Pick the highest-similarity frame as the first candidate.
            2. Expand it to a scene interval via _find_left/_find_right.
            3. Mark the interval as selected; repeat from remaining frames.
            4. Stop when remaining max similarity falls below the threshold.

        Threshold strategy:
            - Initial threshold = max_similarity - similar_threshold.
            - Per-iteration: threshold = current_candidate_similarity - DECAY_STEP (gradual decay).

        Args:
            query_similarities: Per-frame cosine similarity with query (N,).
            image_features: Normalized image features (N, D).
            sample_num: Maximum number of candidate intervals.
            use_threshold: Whether to apply similarity threshold filtering.

        Returns:
            List of intervals, each as [left_boundary, candidate_idx, right_boundary].
        """
        selected_indices = []
        frame_size = len(image_features)
        selected_mask = torch.zeros(frame_size, dtype=torch.bool)

        # Compute similarity floor: max_similarity minus decay
        similar_threshold = 0
        first_idx = torch.argmax(query_similarities).item()
        similar_threshold = query_similarities[first_idx] - self.similar_threshold

        # Expand the highest-similarity frame into the first scene interval
        key_left_id = self._find_left(image_features, first_idx)
        key_right_id = self._find_right(image_features, first_idx)
        selected_indices.append([max(0, key_left_id), first_idx, min(key_right_id, frame_size - 1)])
        selected_mask[key_left_id : key_right_id + 1] = True

        # Greedy iteration: find next best candidate among unselected frames
        cur_similar_threshold = 0
        while selected_mask.sum() < frame_size and len(selected_indices) < sample_num:
            remaining_indices = (~selected_mask).nonzero(as_tuple=True)[0]

            if len(remaining_indices) == 0:
                break
            new_query_similarities = query_similarities[remaining_indices]

            candidate_idx = torch.argmax(new_query_similarities).item()

            # Stop if similarity falls below threshold
            if use_threshold and (
                new_query_similarities[candidate_idx] < similar_threshold
                or new_query_similarities[candidate_idx] < cur_similar_threshold
            ):
                break

            # Decay threshold by DECAY_STEP per iteration
            cur_similar_threshold = new_query_similarities[candidate_idx] - DECAY_STEP
            key_left_id = self._find_left(image_features, remaining_indices[candidate_idx])
            key_right_id = self._find_right(image_features, remaining_indices[candidate_idx])
            selected_indices.append(
                [max(0, key_left_id), int(remaining_indices[candidate_idx]), min(key_right_id, frame_size - 1)]
            )
            selected_mask[key_left_id : key_right_id + 1] = True
        return selected_indices

    def _merge_intervals(
        self, intervals: List[List[int]], threshold: int = 30, query_similarities: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """Merges adjacent semantically similar intervals using Union-Find.

        Merge conditions (both must hold):
            1. Spatial proximity: gap between intervals <= threshold frames.
            2. Semantic similarity: difference in mean query similarity
               between intervals < image_similar_threshold.

        Algorithm:
            1. Sort intervals by left endpoint.
            2. Use a sliding window to skip non-adjacent pairs.
            3. Union intervals meeting both conditions.
            4. Collect connected components and take global min/max endpoints.

        Args:
            intervals: List of intervals, each as [left, candidate, right].
            threshold: Maximum gap (in frames) for two intervals to be considered adjacent.
            query_similarities: Per-frame cosine similarity with query.

        Returns:
            Merged interval list, each as [left, right].
        """
        if not intervals:
            return []

        # Preprocess: extract endpoints, mean similarity, and key-frame set per interval
        processed = []
        for orig_idx, pts in enumerate(intervals):
            sorted_pts = sorted(pts)
            min_pt, max_pt = sorted_pts[0], sorted_pts[-1]
            points_set = set(sorted_pts)

            # Mean query similarity within the interval as semantic similarity measure
            if query_similarities is not None:
                max_sim = query_similarities[min_pt : max_pt + 1].mean().item()
            else:
                max_sim = 1.0

            processed.append((orig_idx, min_pt, max_pt, max_sim, points_set))

        # Sort by left endpoint for sliding-window scan
        processed.sort(key=lambda x: (x[1], x[0]))

        # Union-Find initialization
        n = len(processed)
        parent, rank = list(range(n)), [0] * n

        def find(i: int) -> int:
            """Find with path compression."""
            root = i
            while parent[root] != root:
                root = parent[root]
            while parent[i] != i:
                next_node = parent[i]
                parent[i] = root
                i = next_node
            return root

        def union(i: int, j: int) -> bool:
            """Union by rank."""
            ri, rj = find(i), find(j)
            if ri == rj:
                return False
            if rank[ri] < rank[rj]:
                parent[ri] = rj
            elif rank[ri] > rank[rj]:
                parent[rj] = ri
            else:
                parent[rj] = ri
                rank[ri] += 1
            return True

        # Sliding-window scan: skip non-adjacent pairs via sorted monotonicity
        window_start = 0

        for i in range(n):
            curr_orig, curr_min, curr_max, curr_sim, curr_pts = processed[i]

            # Advance window left edge: skip intervals that cannot be adjacent
            while window_start < i:
                _, _, prev_max, _, _ = processed[window_start]
                if prev_max + threshold < curr_min:
                    window_start += 1
                else:
                    break

            # Check each pair in the window for merge eligibility
            for j in range(window_start, i):
                prev_orig, prev_min, prev_max, prev_sim, prev_pts = processed[j]

                # Condition 1: not spatially adjacent
                if curr_min > prev_max + threshold:
                    continue

                # Condition 2: not semantically similar
                if abs(curr_sim - prev_sim) >= self.image_similar_threshold:
                    continue

                # Both conditions met — merge
                union(curr_orig, prev_orig)

        # Group by connected components
        groups = defaultdict(list)
        for item in processed:
            orig_idx = item[0]
            root = find(orig_idx)
            groups[root].append(item)

        # Take global min/max endpoints per component
        results = []
        for items in groups.values():
            all_points = set()
            global_min, global_max, earliest_orig_idx = float('inf'), -float('inf'), float('inf')

            for orig_idx, min_pt, max_pt, _, pts in items:
                all_points.update(pts)
                global_min = min(global_min, min_pt)
                global_max = max(global_max, max_pt)
                earliest_orig_idx = min(earliest_orig_idx, orig_idx)

            results.append((earliest_orig_idx, global_min, global_max))

        # Sort by original appearance order
        results.sort(key=lambda x: x[0])
        return [[min_v, max_v] for _, min_v, max_v in results]

    def _resample(
        self,
        frame_indices: List[List[int]],
        image_features: torch.Tensor,
        query_feature: torch.Tensor,
        sample_num: int = 16,
    ) -> List[int]:
        """Adaptively resamples across merged intervals, respecting the total frame budget.

        Args:
            frame_indices: Merged intervals, each as [left, right].
            image_features: Normalized image features for all frames.
            query_feature: Normalized query text feature.
            sample_num: Total frame budget.

        Returns:
            List of resampled frame indices across all intervals.
        """
        new_frame_indices = []
        for interval in frame_indices:
            min_idx, max_idx = min(interval), max(interval)

            # Allocate quota proportional to interval length
            sampled = self._adaptive_resample(
                interval, image_features, query_feature, base_sample_num=self._get_sample_num(max_idx - min_idx + 1)
            )

            # Respect remaining budget
            left_sample_num = max(sample_num - len(new_frame_indices), 0)
            new_frame_indices.extend(sampled[:left_sample_num])
            if len(new_frame_indices) >= sample_num:
                break
        return new_frame_indices


class KFrameSelector(BaseFrameSelector):
    """Discrete keyframe selector: selects visually diverse keyframes relevant to the query.

    Unlike KRangFrameSelector which outputs continuous intervals, this selector
    outputs discrete frame indices for tasks requiring only visual diversity.

    Pipeline:
        1. Compute per-frame query similarity.
        2. Greedily select diverse high-similarity candidates.
        3. Pick top-2 frames per scene interval as representatives.
    """

    def select_keyframes(
        self, query: str, frames: List[np.ndarray], sample_num: int, do_resample: bool = False
    ) -> Tuple[List[int], List[np.ndarray]]:
        """Discrete keyframe selection pipeline.

        Args:
            query: Query text.
            frames: Video frame list.
            sample_num: Maximum keyframe count.
            do_resample: Unused in this selector.

        Returns:
            Tuple of (keyframe indices, keyframe images).
        """
        sample_num = self._validate_select_inputs(query, frames, sample_num)

        # Step 1: extract features and compute per-frame similarity
        query_feature, image_features, query_similarities = self._compute_similarities(query, frames)

        # Step 2: greedily select diverse candidates, picking top-2 per scene interval
        frame_indices = self._select_diverse_frames(query_similarities, image_features, sample_num)
        unique_indices = sorted(set(frame_indices))
        key_frames = [frames[i] for i in unique_indices]
        return unique_indices, key_frames

    def _select_diverse_frames(
        self,
        query_similarities: torch.Tensor,
        image_features: torch.Tensor,
        sample_num: int,
        use_threshold: bool = False,
    ) -> List[int]:
        """Greedily selects diverse discrete keyframes relevant to the query.

        Differences from KRangFrameSelector._select_diverse_frames:
            - Selects top-2 frames per scene interval (not the whole interval).
            - Defaults to no threshold filtering (use_threshold=False).
            - Applies Euclidean-distance diversity filtering (min distance >= DIVERSITY_DISTANCE_THRESHOLD).

        Algorithm:
            1. Pick the highest-similarity frame as the first candidate.
            2. Determine its scene interval via _find_left/_find_right.
            3. Select top-2 frames by similarity within the interval.
            4. Mark the interval as selected; repeat from remaining frames.
            5. Each new candidate must pass a diversity filter against already-selected frames.

        Args:
            query_similarities: Per-frame cosine similarity with query (N,).
            image_features: Normalized image features (N, D).
            sample_num: Maximum number of candidate intervals.
            use_threshold: Whether to apply similarity threshold filtering.

        Returns:
            List of selected keyframe indices.
        """
        selected_indices = []
        frame_size = len(image_features)
        selected_mask = torch.zeros(frame_size, dtype=torch.bool)

        # Mean absolute diff of similarity sequence — reference for diversity
        mean_diff = torch.abs(torch.diff(query_similarities)).mean().item()
        mean_diff = round(mean_diff, 4)

        # Pick the highest-similarity frame as first candidate
        first_idx = torch.argmax(query_similarities).item()

        # Determine its scene interval
        key_left_id = self._find_left(image_features, first_idx)
        key_right_id = self._find_right(image_features, first_idx)
        interval_frames = list(range(key_left_id, key_right_id + 1))

        # Select top-2 frames by similarity within the interval
        sims_in_interval = query_similarities[interval_frames]
        _, top_local_idx = torch.topk(sims_in_interval, min(2, len(interval_frames)))
        for local_idx in top_local_idx:
            selected_indices.append(interval_frames[local_idx])

        selected_mask[key_left_id : key_right_id + 1] = True

        # Staircase decay: gradually lower the similarity bar each iteration
        stage_diff = INITIAL_STAGE_DIFF
        cur_similar_threshold = round((query_similarities[first_idx] - stage_diff).item(), 4)
        stage_diff = stage_diff - STAGE_DECAY_STEP

        # Greedy iteration: find next best candidate among unselected frames
        while selected_mask.sum() < frame_size and len(selected_indices) < sample_num:
            remaining_indices = (~selected_mask).nonzero(as_tuple=True)[0]

            if len(remaining_indices) == 0:
                break
            new_query_similarities = query_similarities[remaining_indices]

            candidate_idx = torch.argmax(new_query_similarities).item()

            candidate_sim = round(new_query_similarities[candidate_idx].item(), 4)

            # Stop if similarity falls below current staircase threshold
            if use_threshold and (candidate_sim < cur_similar_threshold):
                break

            # Decay staircase threshold, floor at mean_diff
            cur_similar_threshold = round((new_query_similarities[candidate_idx] - stage_diff).item(), 4)
            stage_diff = max(mean_diff, stage_diff - STAGE_DECAY_STEP)

            # Determine candidate's scene interval
            key_left_id = self._find_left(image_features, remaining_indices[candidate_idx])
            key_right_id = self._find_right(image_features, remaining_indices[candidate_idx])

            # Diversity filter: only add if min Euclidean distance to selected frames >= DIVERSITY_DISTANCE_THRESHOLD
            remaining_features = image_features[remaining_indices[candidate_idx]].unsqueeze(0)
            selected_features = image_features[selected_indices]
            distances = torch.cdist(remaining_features, selected_features)
            min_distances, _ = torch.min(distances, dim=1)
            max_min_dist = torch.max(min_distances).item()
            if max_min_dist >= DIVERSITY_DISTANCE_THRESHOLD:
                selected_indices.append(int(remaining_indices[candidate_idx]))

            # Always mark the interval as selected to avoid revisiting
            selected_mask[key_left_id : key_right_id + 1] = True

        return selected_indices
