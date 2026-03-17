import os
import logging
import pytest
from mm import load_audio
from mm_test.common import TEST_HW_USER_FILE_PATH, TEST_HW_USER_WAV_LOAD_PATH

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_audio_func(wav_name, sampling_fraction):
    if isinstance(wav_name, str) and wav_name.endswith('.wav'):
        wav_inputs = os.path.join(TEST_HW_USER_WAV_LOAD_PATH, wav_name)
        logger.info(f'Loading audio file: {wav_inputs}')
        audio, sr = load_audio(wav_inputs, sampling_fraction)
        mm_numpydata = audio.numpy()
        logger.info(
            f'Audio loaded successfully - shape: {mm_numpydata.shape}, sampling rate: {sr}, size: {mm_numpydata.size}')
        return mm_numpydata, sr
    else:
        if isinstance(wav_name, list):
            wav_inputs = []
            for wav in wav_name:
                wav_input = os.path.join(TEST_HW_USER_WAV_LOAD_PATH, wav)
                wav_inputs.append(wav_input)
            logger.info(f'Loading audio files: {wav_inputs}')
        elif isinstance(wav_name, str):
            wav_inputs = os.path.join(TEST_HW_USER_FILE_PATH, wav_name)
        else:
            raise TypeError("Only str or list types are supported")
        
        audio_list_load = load_audio(wav_inputs, sampling_fraction)
        results = []
        for i, audio in enumerate(audio_list_load):
            logger.info(f'Processing audio result {i+1}/{len(audio_list_load)}')
            test_audio = audio[0].numpy()
            test_sr = audio[1]
            logger.info(
                f'Audio {i+1} - shape: {test_audio.shape}, sampling rate: {test_sr}')
            if sampling_fraction:
                assert sampling_fraction == test_sr
            results.append((test_audio, test_sr))
        return results


def test_wav_file_sr_16000():
    wav_name = 'audio_test0.wav'
    sampling_fraction = 16000
    mm_numpydata, sr = load_audio_func(wav_name, sampling_fraction)
    assert sr == sampling_fraction, f"Sampling rate mismatch: expected {sampling_fraction}, got {sr}"
    assert mm_numpydata.size > 0, "Audio data should not be empty"
    

def test_wav_list_sr_16000():
    wav_list = ['audio_test0.wav', 'audio_test1.wav']
    sampling_fraction = 16000
    results = load_audio_func(wav_list, sampling_fraction)
    assert len(results) == len(wav_list), f"Result count mismatch: expected {len(wav_list)}, got {len(results)}"
    for i, (audio_data, sr) in enumerate(results):
        assert sr == sampling_fraction, f"Sampling rate mismatch for audio {i}: expected {sampling_fraction}, got {sr}"
        assert audio_data.size > 0, f"Audio data {i} should not be empty"


def test_wav_dir_sr_16000():
    wav_dir = 'wav_load_test'
    sampling_fraction = 16000
    results = load_audio_func(wav_dir, sampling_fraction)
    assert len(results) > 0, "Should have loaded at least one audio file from directory"
    for i, (audio_data, sr) in enumerate(results):
        assert sr == sampling_fraction, f"Sampling rate mismatch for audio {i}: expected {sampling_fraction}, got {sr}"
        assert audio_data.size > 0, f"Audio data {i} should not be empty"
