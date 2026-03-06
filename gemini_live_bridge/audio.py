"""Audio conversion utilities for the voice bridge."""

import numpy as np


def convert_32bit_stereo_to_16bit_mono(data: bytes) -> bytes:
    """Convert 32-bit stereo I2S to 16-bit mono PCM (extract XMOS AEC channel).

    ESP32 sends stereo 32-bit I2S at 16kHz. Left channel (ch0) contains
    XMOS AEC-processed audio, right channel (ch1) is silent.
    """
    if len(data) % 4 != 0:
        return data
    all_samples = np.frombuffer(data, dtype='<i4')
    ch0 = all_samples[0::2]  # left = XMOS AEC processed
    return (ch0 >> 16).astype('<i2').tobytes()


def resample_16k_to_24k(data: bytes) -> bytes:
    """Resample 16-bit PCM from 16kHz to 24kHz via linear interpolation.

    Used by OpenAI Realtime provider which expects 24kHz input.
    """
    samples = np.frombuffer(data, dtype='<i2').astype(np.float32)
    if len(samples) < 2:
        return data
    n_out = int(len(samples) * 1.5)  # 24000/16000 = 1.5
    x_old = np.linspace(0, 1, len(samples))
    x_new = np.linspace(0, 1, n_out)
    resampled = np.interp(x_new, x_old, samples)
    return np.clip(resampled, -32768, 32767).astype('<i2').tobytes()


def process_gemini_audio(data: bytes) -> bytes:
    """Convert Gemini 24kHz 16-bit mono to device 48kHz 32-bit stereo.

    NOT used in current pipeline (ESP32 converts locally), kept for reference.
    """
    samples = np.frombuffer(data, dtype='<i2')
    if len(samples) < 1:
        return data
    return (np.repeat(samples.astype(np.int32), 4) << 16).tobytes()
