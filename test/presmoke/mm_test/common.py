import os.path

# Demo file paths used for testing (for regular user HwHiAiUser)
TEST_HW_USER_FILE_PATH = "/workspace/presmoke_data/multimodal/doc"
TEST_HW_USER_IMAGES_PATH = os.path.join(TEST_HW_USER_FILE_PATH, "images")
TEST_HW_USER_VIDEO_PATH = os.path.join(TEST_HW_USER_FILE_PATH, "ge_videos")
TEST_HW_USER_WAV_NOISE_PATH = os.path.join(TEST_HW_USER_FILE_PATH, "wav_noise_test")
TEST_HW_USER_WAV_LOAD_PATH = os.path.join(TEST_HW_USER_FILE_PATH, "wav_load_test")

# CPU side device variable
DEVICE_CPU = "cpu"

# ---------------------------------------------------------------------------
# mm_e2e (vLLM) pre-smoke test parameters
# ---------------------------------------------------------------------------
MODEL_PATHS = [
    "/home/models/Qwen3-VL-8B-Instruct",
]

IMAGE_PATH = os.path.join(TEST_HW_USER_IMAGES_PATH, "img_1024x1920.jpeg")

VIDEO_PATH = os.path.join(TEST_HW_USER_VIDEO_PATH, "480p_10fps_300s.mp4")

# vLLM OpenAI-compatible server settings.
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 18000

# MM_SCC_RATE forwarded to the vLLM subprocess. Values < 1.0 enable the SDK
# SCC token compression patches; 1.0 disables them.
MM_SCC_RATE = "0.5"

# Wait at most this many seconds for the vLLM server to become ready.
VLLM_SERVER_READY_TIMEOUT = 600

# Poll interval (seconds) when probing /v1/models for readiness.
VLLM_SERVER_READY_INTERVAL = 5

# Extra CLI flags appended to `vllm serve`. Keep this empty by default; tune
# for your hardware (e.g. `--gpu-memory-utilization`).
VLLM_EXTRA_ARGS: list = []

# Prompt template used for both image and video inference calls.
USER_PROMPT = "Describe what you see in the media."
