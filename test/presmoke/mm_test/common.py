import os.path

# Demo file paths used for testing (for regular user HwHiAiUser)
TEST_HW_USER_FILE_PATH = "/home/presmoke_data/multimodal/doc"
TEST_HW_USER_IMAGES_PATH = os.path.join(TEST_HW_USER_FILE_PATH, "images")
TEST_HW_USER_VIDEO_PATH = os.path.join(TEST_HW_USER_FILE_PATH, "ge_videos")
TEST_HW_USER_WAV_NOISE_PATH = os.path.join(TEST_HW_USER_FILE_PATH, "wav_noise_test")
TEST_HW_USER_WAV_LOAD_PATH = os.path.join(TEST_HW_USER_FILE_PATH, "wav_load_test")

# CPU side device variable
DEVICE_CPU = "cpu"