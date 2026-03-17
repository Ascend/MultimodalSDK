import os
import time
import logging
import pytest
import librosa
from mm import load_audio
from mm_test.common import TEST_HW_USER_WAV_LOAD_PATH
import numpy as np

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def accuracy_comparison(audio_files_path, sampling_fraction):
    audio_files = []
    errors = []
    for root, dirs, files in os.walk(audio_files_path):
        for file in files:
            audio_files.append(os.path.join(root, file))
    if len(audio_files) > 128:
        audio_files = [f for f in audio_files if f.split('/')[-1].startswith('audio_test')]
    for audio_file in audio_files:
        audio, sr = load_audio(audio_file, sampling_fraction)
        mm_numpydata = audio.numpy()
        y_librosa, sr_librosa = librosa.load(audio_file, sr=sampling_fraction, mono=True, res_type="soxr_hq")
        try:
            assert sr == sr_librosa, f'load_audio {sr} != librosa.load {sr_librosa}'
            assert np.allclose(mm_numpydata, y_librosa, rtol=1e-4, atol=1e-4), \
                f"Arrays differ! Max difference = {np.abs(mm_numpydata - y_librosa).max()}"
        except AssertionError as e:
            logger.error(f"{audio_file}: Failed -> {e}")
            errors.append(str(e))
    if errors:
        pytest.fail(f"error count: {len(errors)}")


def test_load_audio_acc():
    logger.info("Starting test_load_audio_acc with sampling rate 16000")
    accuracy_comparison(TEST_HW_USER_WAV_LOAD_PATH, sampling_fraction=16000)
    logger.info("Completed test_load_audio_acc with sampling rate 16000 successfully")
    
    logger.info("Starting test_load_audio_acc with sampling rate 44100")
    accuracy_comparison(TEST_HW_USER_WAV_LOAD_PATH, sampling_fraction=44100)
    logger.info("Completed test_load_audio_acc with sampling rate 44100 successfully")


def test_load_audio_perf():
    audio_files = []
    for root, dirs, files in os.walk(TEST_HW_USER_WAV_LOAD_PATH):
        for file in files:
            audio_files.append(os.path.join(root, file))
    if len(audio_files) > 128:
        audio_files = [f for f in audio_files if f.split('/')[-1].startswith('audio_test')]
    durations = []
    
    logger.info("Testing load_audio performance for single wave files")
    for audio_file in audio_files:
        start = time.time()
        load_audio(audio_file, 16000)
        end = time.time()
        duration = end - start
        durations.append(duration)
    average_time = sum(durations) / len(audio_files)
    logger.info(f"Single wave file average time: {average_time:.4f}s")
    
    logger.info("Testing load_audio performance for wave file list")
    start = time.time()
    load_audio(audio_files, 16000)
    end = time.time()
    duration = end - start
    logger.info(f"Wave file list total time: {duration:.4f}s, single file average: {average_time:.4f}s")
    
    logger.info("Testing load_audio performance for wave file directory")
    start = time.time()
    load_audio(TEST_HW_USER_WAV_LOAD_PATH, 16000)
    end = time.time()
    duration = end - start
    logger.info(f"Wave file directory total time: {duration:.4f}s, single file average: {average_time:.4f}s")