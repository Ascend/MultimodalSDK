#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. See http://license.coscl.org.cn/MulanPSL2
"""CPU contract tests with vLLM kernels and sibling algorithms replaced by doubles."""

import importlib.util
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch


SOURCE = Path(__file__).resolve().parents[1] / 'source/mm/patcher/vllm/flashvid/patch_qwen2_5_vl.py'


def load_patch():
    names = (
        'vllm.config',
        'vllm.distributed',
        'vllm.logger',
        'vllm.model_executor.layers.attention',
        'vllm.model_executor.models.interfaces',
        'vllm.model_executor.models.qwen2_5_vl',
        'vllm.model_executor.models.utils',
        'vllm.model_executor.models.vision',
        'vllm.multimodal.evs',
        'vllm.multimodal.processing',
        'qwen25_test',
        'qwen25_test.method',
        'qwen25_test.token_selection',
    )
    modules = {name: ModuleType(name) for name in names}
    modules['vllm.config'].VllmConfig = object
    modules['vllm.distributed'].tensor_model_parallel_all_reduce = Mock()
    modules['vllm.logger'].init_logger = logging.getLogger
    modules['vllm.model_executor.layers.attention'].MMEncoderAttention = NS(
        compute_max_seqlen=lambda backend, offsets: int(max(offsets[1:] - offsets[:-1])),
        maybe_compute_seq_lens=lambda backend, offsets, device: torch.tensor(offsets[1:] - offsets[:-1], device=device),
    )
    modules['vllm.model_executor.models.interfaces'].MultiModalEmbeddings = tuple
    models = modules['vllm.model_executor.models.qwen2_5_vl']
    for name in (
        'Qwen2_5_VisionAttention',
        'Qwen2_5_VLForConditionalGeneration',
        'Qwen2_5_VisionBlock',
        'Qwen2_5_VisionTransformer',
        'Qwen2_5_VLMultiModalProcessor',
    ):
        setattr(models, name, type(name, (), {'_get_prompt_updates': Mock(return_value='upstream')}))
    models.Qwen2_5_VLVideoInputs = dict
    modules['vllm.model_executor.models.utils'].cast_overflow_tensors = lambda x: x
    modules['vllm.model_executor.models.vision'].run_dp_sharded_mrope_vision_model = Mock()
    modules['vllm.multimodal.evs'].compute_mrope_for_media = Mock()
    modules['vllm.multimodal.processing'].PromptReplacement = NS
    for name in ('compute_flashvid_token_budget', 'flashvid_compression', 'get_prune_config'):
        setattr(modules['qwen25_test.method'], name, Mock())
    for name in ('prune_embeds', 'restore_hidden_states'):
        setattr(modules['qwen25_test.token_selection'], name, Mock())
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('qwen25_test.patch', SOURCE)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
    return result


class TestQwen25FlashVid(unittest.TestCase):
    def setUp(self):
        self.module = load_patch()
        self.config = NS(enable_flashvid=True, enable_encode_prune=False, retention_ratio=0.5, alpha=0.7)

    def vision(self):
        calls = []

        def block(x, **kwargs):
            calls.append(kwargs)
            return x, torch.arange(len(x), dtype=torch.float32) if kwargs['return_attn_score'] else None

        metadata = dict(
            rotary_pos_emb_cos=torch.ones(16, 2),
            rotary_pos_emb_sin=torch.zeros(16, 2),
            window_index=torch.tensor([2, 0, 3, 1]),
            reverse_indices=torch.tensor([1, 3, 0, 2]),
            cu_seqlens=torch.tensor([0, 8, 16], dtype=torch.int32),
            cu_window_seqlens=torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32),
            max_seqlen_full=torch.tensor(8),
            max_seqlen_window=torch.tensor(4),
            sequence_lengths_full=torch.tensor([8, 8]),
            sequence_lengths_window=torch.tensor([4] * 4),
        )
        visual = NS(
            device='cpu',
            dtype=torch.float32,
            patch_embed=lambda x: x,
            spatial_merge_unit=4,
            prepare_encoder_metadata=lambda grid: metadata,
            blocks=[block, block],
            fullatt_block_indexes=[1],
            merger=lambda x: x.reshape(-1, 4, 2).mean(1),
            attn_backend='test',
        )
        return visual, calls, metadata

    def test_image_and_disabled_forward_return_tensor_without_scores(self):
        visual, calls, _ = self.vision()
        output = self.module._qwen2_5_vision_transformer_forward(visual, torch.ones(16, 2), [[2, 2, 4]])
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (4, 2))
        self.assertEqual([call['return_attn_score'] for call in calls], [False, False])

    def test_only_last_layer_scores_are_grouped_and_reordered(self):
        visual, calls, _ = self.vision()
        _, scores = self.module._qwen2_5_vision_transformer_forward(
            visual, torch.ones(16, 2), [[2, 2, 4]], return_attn_score=True
        )
        self.assertEqual([call['return_attn_score'] for call in calls], [False, True])
        torch.testing.assert_close(scores, torch.tensor([5.5, 13.5, 1.5, 9.5]))

    def test_pruning_refreshes_full_and_window_backend_lengths(self):
        visual, calls, metadata = self.vision()
        self.module.prune_embeds.side_effect = lambda x: (torch.arange(0, len(x), 2), torch.arange(1, len(x), 2))
        self.module.restore_hidden_states.side_effect = lambda x, length, keep, removed, score: (
            torch.zeros(length, 1, 2),
            None,
        )
        output = self.module._qwen2_5_vision_transformer_forward(
            visual, torch.ones(16, 2), [[2, 2, 4]], prune_encoder=True
        )
        self.assertEqual(output.shape, (4, 2))
        for call, lengths in zip(calls, ([2, 2, 2, 2], [4, 4])):
            self.assertEqual(call['sequence_lengths'].tolist(), lengths)
            self.assertEqual(call['max_seqlen'].item(), max(lengths))
            self.assertEqual(call['cu_seqlens'][-1].item(), 8)
        self.assertEqual(metadata['cu_seqlens'].tolist(), [0, 8, 16])

    def model(self, enabled=True, dp=False):
        return NS(
            prune_config=NS(**{**vars(self.config), 'enable_flashvid': enabled}),
            use_data_parallel=dp,
            visual=Mock(dtype=torch.float32, spatial_merge_size=2),
            config=NS(vision_config=NS(tokens_per_second=25)),
        )

    def video(self):
        return dict(
            type='pixel_values_videos',
            pixel_values_videos=torch.ones(8, 2),
            video_grid_thw=torch.tensor([[2, 2, 2]]),
            second_per_grid_ts=torch.tensor([0.08]),
        )

    def test_disabled_video_accepts_tensor_output(self):
        model = self.model(False)
        model.visual.return_value = torch.ones(2, 3)
        videos, scores = self.module._qwen2_5_process_video_input(model, self.video())
        self.assertEqual(len(videos), 1)
        self.assertIsNone(scores)
        self.assertFalse(model.visual.call_args.kwargs['return_attn_score'])

    def test_enabled_video_splits_embeddings_and_scores(self):
        model = self.model()
        model.visual.return_value = (torch.ones(2, 3), torch.tensor([0.1, 0.9]))
        videos, scores = self.module._qwen2_5_process_video_input(model, self.video())
        self.assertEqual(videos[0].shape, (2, 3))
        self.assertEqual(scores[0].shape, (2,))

    def test_unsupported_paths_fail_explicitly_only_when_enabled(self):
        for dp, kind, message in (
            (True, 'pixel_values_videos', 'data-parallel'),
            (False, 'video_embeds', 'precomputed'),
        ):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, message):
                self.module._qwen2_5_process_video_input(self.model(dp=dp), {**self.video(), 'type': kind})
        model = self.model(False, True)
        self.module.run_dp_sharded_mrope_vision_model.return_value = (torch.ones(2, 3),)
        videos, scores = self.module._qwen2_5_process_video_input(model, self.video())
        self.assertEqual(len(videos), 1)
        self.assertIsNone(scores)
        videos, _ = self.module._qwen2_5_process_video_input(
            self.model(False), {**self.video(), 'type': 'video_embeds', 'video_embeds': torch.ones(2, 3)}
        )
        self.assertEqual(videos[0].shape, (2, 3))

    def test_missing_metadata_and_mismatched_counts_fail_clearly(self):
        video, embeds = self.video(), (torch.ones(2, 3),)
        for times, scores, message in (
            (None, (torch.ones(2),), 'second_per_grid_ts'),
            (torch.tensor([0.08]), None, 'attention scores'),
            (torch.tensor([0.08]), (), 'matching counts'),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.module._postprocess_video_embeds_flashvid(
                    self.model(), embeds, scores, {**video, 'second_per_grid_ts': times}
                )
        with self.assertRaisesRegex(ValueError, 'grid token count'):
            self.module._postprocess_video_embeds_flashvid(self.model(), embeds, (torch.ones(1),), video)

    def test_fractional_time_and_selected_positions_are_preserved(self):
        self.module.flashvid_compression.return_value = (torch.tensor([[9.0, 8.0, 7.0]]), torch.tensor([1]))
        self.module.compute_mrope_for_media.return_value = torch.tensor([[0, 0, 0, 1], [2, 0, 0, 1]])
        output = self.module._postprocess_video_embeds_flashvid(
            self.model(), (torch.ones(2, 3),), (torch.ones(2),), self.video()
        )
        self.assertAlmostEqual(self.module.compute_mrope_for_media.call_args.kwargs['video_second_per_grid'], 0.08)
        torch.testing.assert_close(output[0], torch.tensor([[9.0, 8.0, 7.0, 2.0, 0.0, 0.0, 1.0]]))

    def test_full_retention_skips_selection_for_single_token_frames(self):
        self.module.compute_flashvid_token_budget.return_value = NS(total_tokens=2)
        self.module.compute_mrope_for_media.return_value = torch.zeros(2, 4)
        embeddings = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        output = self.module._postprocess_video_embeds_flashvid(
            self.model(), (embeddings,), (torch.ones(2),), self.video()
        )
        self.module.flashvid_compression.assert_not_called()
        torch.testing.assert_close(output[0][:, :3], embeddings)

    def test_flashvid_dispatch_does_not_depend_on_evs_flag(self):
        model = self.model()
        model.is_multimodal_pruning_enabled = False
        model._parse_and_validate_multimodal_inputs = lambda **kwargs: {'video': self.video()}
        model._process_video_input = lambda video: ((torch.ones(2, 3),), (torch.ones(2),))
        model._postprocess_video_embeds_flashvid = Mock(return_value=(torch.ones(1, 7),))
        output = self.module._embed_multimodal(model)
        self.assertEqual(output[0].shape, (1, 7))
        model._postprocess_video_embeds_flashvid.assert_called_once()

    def test_processor_uses_shared_budget_and_leaves_images_uncompressed(self):
        self.module.get_prune_config.return_value = self.config
        self.module.compute_flashvid_token_budget.return_value = NS(total_tokens=3)
        processor = NS(
            info=NS(
                ctx=NS(get_mm_config=lambda: NS(video_pruning_rate=0.5)),
                get_hf_processor=lambda **kw: NS(image_token='image', video_token='video'),
                get_image_processor=lambda **kw: NS(merge_size=2),
                get_tokenizer=lambda: NS(get_vocab=lambda: {'image': 1, 'video': 2}),
            )
        )
        outputs = {kind: [{f'{kind}_grid_thw': NS(data=torch.tensor([2, 2, 4]))}] for kind in ('image', 'video')}
        updates = self.module._get_prompt_updates(processor, None, {}, outputs)
        self.assertEqual(updates[0].replacement(0), [1] * 4)
        self.assertEqual(updates[1].replacement(0), [2] * 3)
        self.module.compute_flashvid_token_budget.assert_called_once_with(2, 2, 0.5, 0.7)
        self.config.enable_flashvid = False
        self.assertEqual(self.module._get_prompt_updates(processor, None, {}, outputs), 'upstream')


if __name__ == '__main__':
    unittest.main()
