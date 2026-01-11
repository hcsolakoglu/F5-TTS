#!/usr/bin/env python3
"""
Build test audio pack for integration tests.

This script creates a small audio test pack for running integration tests.
It supports two modes:

Mode A (streaming): Download samples from HuggingFace datasets
Mode B (local): Use user-provided audio files

Usage:
    # Mode A - streaming from HuggingFace
    python scripts/build_test_audio_pack.py --mode streaming --num-samples 5

    # Mode B - local files
    python scripts/build_test_audio_pack.py --mode local --input-dir /path/to/audio

Output is written to tests/assets/audio_pack/
"""

import argparse
import json
import os
import sys
from pathlib import Path


def build_streaming_pack(output_dir: Path, num_samples: int = 5):
    """Build test pack from HuggingFace datasets streaming."""
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        print("ERROR: datasets and soundfile are required for streaming mode.")
        print("Install with: pip install datasets soundfile")
        sys.exit(1)

    print(f"Streaming {num_samples} samples from LibriSpeech...")

    # Use streaming to avoid downloading full dataset
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split="test",
        streaming=True,
        trust_remote_code=True,
    )

    manifest = []
    count = 0

    for sample in ds:
        if count >= num_samples:
            break

        audio = sample["audio"]
        text = sample["text"]
        speaker_id = sample.get("speaker_id", f"spk_{count}")

        # Save audio
        audio_path = output_dir / f"sample_{count:03d}.wav"
        sf.write(
            audio_path,
            audio["array"],
            audio["sampling_rate"],
        )

        # Save text
        text_path = output_dir / f"sample_{count:03d}.txt"
        with open(text_path, "w") as f:
            f.write(text)

        manifest.append({
            "id": f"sample_{count:03d}",
            "audio": str(audio_path.name),
            "text": text,
            "speaker_id": str(speaker_id),
            "sample_rate": audio["sampling_rate"],
        })

        count += 1
        print(f"  Saved sample {count}/{num_samples}")

    # Write manifest
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")

    print(f"Manifest written to: {manifest_path}")
    return manifest


def build_local_pack(output_dir: Path, input_dir: Path, target_sr: int = 24000, max_duration: float = 10.0):
    """Build test pack from local audio files."""
    try:
        import torchaudio
        import torch
    except ImportError:
        print("ERROR: torchaudio is required for local mode.")
        sys.exit(1)

    print(f"Processing local files from: {input_dir}")

    # Find audio files
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))
        audio_files.extend(input_dir.glob(f"**/*{ext}"))

    if not audio_files:
        print(f"ERROR: No audio files found in {input_dir}")
        sys.exit(1)

    manifest = []

    for i, audio_path in enumerate(audio_files):
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        # Trim to max duration
        max_samples = int(max_duration * target_sr)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Save normalized audio
        out_name = f"sample_{i:03d}.wav"
        out_path = output_dir / out_name
        torchaudio.save(out_path, waveform, target_sr)

        # Look for corresponding text file
        text_file = audio_path.with_suffix(".txt")
        if text_file.exists():
            with open(text_file) as f:
                text = f.read().strip()
        else:
            text = ""

        # Save text
        out_text_path = output_dir / f"sample_{i:03d}.txt"
        with open(out_text_path, "w") as f:
            f.write(text)

        manifest.append({
            "id": f"sample_{i:03d}",
            "audio": out_name,
            "text": text,
            "speaker_id": audio_path.stem,
            "sample_rate": target_sr,
            "original_file": str(audio_path),
        })

        print(f"  Processed: {audio_path.name}")

    # Write manifest
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")

    print(f"Manifest written to: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build test audio pack for integration tests"
    )
    parser.add_argument(
        "--mode",
        choices=["streaming", "local"],
        default="streaming",
        help="Mode: 'streaming' from HF datasets, or 'local' from user files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/assets/audio_pack",
        help="Output directory for the audio pack",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory for local mode",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to include (streaming mode)",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=24000,
        help="Target sample rate",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Maximum audio duration in seconds",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "streaming":
        manifest = build_streaming_pack(output_dir, args.num_samples)
    else:
        if not args.input_dir:
            print("ERROR: --input-dir is required for local mode")
            sys.exit(1)
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            sys.exit(1)
        manifest = build_local_pack(
            output_dir, input_dir, args.target_sr, args.max_duration
        )

    # Write README
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("# Test Audio Pack\n\n")
        f.write("This directory contains audio samples for integration testing.\n\n")
        f.write("## Contents\n\n")
        f.write(f"- {len(manifest)} audio samples\n")
        f.write("- `manifest.jsonl`: Metadata for all samples\n\n")
        f.write("## License\n\n")
        if args.mode == "streaming":
            f.write("Audio samples are from LibriSpeech ASR corpus.\n")
            f.write("LibriSpeech is released under CC BY 4.0 license.\n")
            f.write("See: https://www.openslr.org/12/\n")
        else:
            f.write("Audio samples are from user-provided files.\n")
            f.write("Please ensure you have appropriate rights to use these files.\n")

    print(f"\nAudio pack created at: {output_dir}")
    print(f"Total samples: {len(manifest)}")


if __name__ == "__main__":
    main()
