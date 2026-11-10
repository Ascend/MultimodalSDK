# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. See http://license.coscl.org.cn/MulanPSL2
"""CPU adapter contract tests; vLLM kernels and sibling algorithms are doubles."""

import importlib.util
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch


def load_patch():
    names = (
        'vllm.config',
        'vllm.logger',
        'vllm.model_executor.layers.attention',
        'vllm.model_executor.models.interfaces',
        'vllm.model_executor.models.qwen2_5_vl',
        'vllm.model_executor.models.qwen3_vl',
        'vllm.model_executor.models.vision',
        'qwen3_test.method',
        'qwen3_test.token_selection',
    )
    modules = {name: ModuleType(name) for name in names}
    modules['vllm.config'].VllmConfig = object
    modules['vllm.logger'].init_logger = logging.getLogger
    modules['vllm.model_executor.layers.attention'].MMEncoderAttention = NS()
    modules['vllm.model_executor.models.interfaces'].MultiModalEmbeddings = tuple
    modules['vllm.model_executor.models.qwen2_5_vl'].Qwen2_5_VLVideoInputs = dict
    for name in ('Qwen3VLForConditionalGeneration', 'Qwen3_VisionBlock', 'Qwen3_VisionTransformer'):
        setattr(modules['vllm.model_executor.models.qwen3_vl'], name, type(name, (), {}))
    modules['vllm.model_executor.models.vision'].run_dp_sharded_mrope_vision_model = Mock()
    for name in ('flashvid_compression', 'get_prune_config'):
        setattr(modules['qwen3_test.method'], name, Mock())
    for name in ('prune_embeds', 'restore_hidden_states'):
        setattr(modules['qwen3_test.token_selection'], name, Mock())
    source = Path(__file__).resolve().parents[1] / 'source/mm/patcher/vllm/flashvid/patch_qwen3_vl.py'
    spec = importlib.util.spec_from_file_location('qwen3_test.patch', source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class TestQwen3Contracts(unittest.TestCase):
    def setUp(self):
        self.module = load_patch()
        self.model = NS(
            prune_config=NS(enable_flashvid=False, enable_encode_prune=False, retention_ratio=0.5),
            visual=Mock(dtype=torch.float32, spatial_merge_size=2),
            use_data_parallel=False,
        )
        self.video = dict(
            type='pixel_values_videos',
            pixel_values_videos=torch.ones(8, 2),
            video_grid_thw=torch.tensor([[2, 2, 2]]),
            timestamps=[[0.0, 0.5]],
        )

    def test_disabled_pixel_output_remains_two_tuple_even_with_two_tokens(self):
        embeddings = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.model.visual.return_value = embeddings
        videos, scores = self.module._process_video_input(self.model, self.video)
        self.assertIsNone(scores)
        torch.testing.assert_close(videos[0], embeddings)

    def test_precomputed_embeddings_return_two_tuple(self):
        video = {**self.video, 'type': 'video_embeds', 'video_embeds': torch.ones(2, 3)}
        videos, scores = self.module._process_video_input(self.model, video)
        self.assertEqual(videos[0].shape, (2, 3))
        self.assertIsNone(scores)
        self.model.prune_config.enable_flashvid = True
        with self.assertRaisesRegex(ValueError, 'precomputed'):
            self.module._process_video_input(self.model, video)

    def test_enabled_pixel_output_splits_scores(self):
        self.model.prune_config.enable_flashvid = True
        self.model.visual.return_value = (torch.ones(2, 3), torch.ones(2))
        videos, scores = self.module._process_video_input(self.model, self.video)
        self.assertEqual(videos[0].shape, (2, 3))
        self.assertEqual(scores[0].shape, (2,))

    def test_data_parallel_rejected_only_when_flashvid_enabled(self):
        self.model.use_data_parallel = True
        self.module.run_dp_sharded_mrope_vision_model.return_value = (torch.ones(2, 3),)
        videos, scores = self.module._process_video_input(self.model, self.video)
        self.assertEqual(len(videos), 1)
        self.assertIsNone(scores)
        self.model.prune_config.enable_flashvid = True
        with self.assertRaisesRegex(ValueError, 'data-parallel'):
            self.module._process_video_input(self.model, self.video)

    def test_missing_scores_timestamps_and_truncated_lists_fail_explicitly(self):
        for video, scores, message in (
            (self.video, None, 'attention scores'),
            ({**self.video, 'timestamps': None}, (torch.ones(2),), 'timestamps'),
            (self.video, (), 'matching counts'),
            ({**self.video, 'timestamps': [[0.0]]}, (torch.ones(2),), 'number of video frames'),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.module._postprocess_video_embeds_flashvid(self.model, (torch.ones(2, 3),), scores, video)

    def test_dictionary_timestamps_reach_final_embeddings(self):
        self.module.flashvid_compression.return_value = (torch.ones(1, 3), torch.tensor([1]))
        self.model._create_final_video_embeddings = Mock(return_value=torch.ones(1, 7))
        outputs = self.module._postprocess_video_embeds_flashvid(
            self.model, (torch.ones(2, 3),), (torch.ones(2),), self.video
        )
        self.assertEqual(outputs[0].shape, (1, 7))
        kwargs = self.model._create_final_video_embeddings.call_args.kwargs
        self.assertEqual(kwargs['timestamps'], [0.0, 0.5])
        self.assertEqual(kwargs['num_tokens_per_frame'], [0, 1])
        self.assertEqual(kwargs['retention_mask'].tolist(), [False, True])

    def test_disabled_dispatch_has_no_stdout(self):
        self.model.is_multimodal_pruning_enabled = False
        self.model._parse_and_validate_multimodal_inputs = lambda **kw: {'video': self.video}
        self.model._process_video_input = lambda video: ((torch.ones(2, 3),), None)
        with patch('builtins.print') as output:
            result = self.module._embed_multimodal(self.model)
        output.assert_not_called()
        self.assertEqual(result[0].shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
