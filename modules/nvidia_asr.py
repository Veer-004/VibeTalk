"""
nvidia_asr.py — Speech-to-text via NVIDIA's hosted Parakeet 1.1B RNNT
Multilingual ASR model (NVCF-hosted Riva endpoint on build.nvidia.com).

Used by vibe_talk_arena_app.py for its voice-input flow.
"""

import os

import riva.client

_SERVER = "grpc.nvcf.nvidia.com:443"
# NVCF function-id for nvidia/parakeet-1_1b-rnnt-multilingual-asr
_FUNCTION_ID = "71203149-d3b7-4460-8231-1be2543a1fca"

_auth = riva.client.Auth(
    uri=_SERVER,
    use_ssl=True,
    metadata_args=[
        ["function-id", _FUNCTION_ID],
        ["authorization", f"Bearer {os.getenv('NVIDIA_API_KEY')}"],
    ],
)
_asr_service = riva.client.ASRService(_auth)


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe a WAV audio clip to text using NVIDIA Parakeet."""
    config = riva.client.RecognitionConfig(
        language_code="multi",
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )
    response = _asr_service.offline_recognize(audio_bytes, config)
    return "".join(
        result.alternatives[0].transcript
        for result in response.results
        if result.alternatives
    ).strip()
