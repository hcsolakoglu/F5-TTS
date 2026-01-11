#!/usr/bin/env python3
"""
Fetch script for ASR reward model (FunASR SenseVoice).

Downloads and caches the SenseVoice model for ASR-based WER reward computation.

Usage:
    python scripts/fetch_reward_asr_model.py [--output-dir DIR] [--cache-dir DIR] [--offline]
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Fetch ASR model for reward computation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save/verify model files",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for model weights (uses default if not specified)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline mode - only use existing cache",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="iic/SenseVoiceSmall",
        help="FunASR model ID to fetch",
    )
    args = parser.parse_args()

    # Check if funasr is installed
    try:
        from funasr import AutoModel
    except ImportError:
        print("ERROR: FunASR is not installed.")
        print("Install it with: pip install 'f5-tts[reward_funasr]'")
        print("Or: pip install funasr")
        sys.exit(1)

    # Set cache directory if specified
    if args.cache_dir:
        os.environ["FUNASR_CACHE"] = args.cache_dir

    if args.offline:
        os.environ["FUNASR_OFFLINE"] = "1"

    print(f"Fetching ASR model: {args.model_id}")

    try:
        # This will download and cache the model
        model_kwargs = {
            "model": args.model_id,
            "disable_update": args.offline,
        }

        if args.output_dir:
            model_kwargs["model"] = args.output_dir

        model = AutoModel(**model_kwargs)

        # Get cache path
        if args.output_dir:
            cache_path = args.output_dir
        elif args.cache_dir:
            cache_path = args.cache_dir
        else:
            cache_path = os.environ.get("FUNASR_CACHE", "~/.cache/funasr")

        print(f"Model fetched successfully!")
        print(f"Cache path: {os.path.expanduser(cache_path)}")

        # Test the model
        print("Testing model with dummy input...")
        import numpy as np

        dummy_audio = np.random.randn(16000).astype(np.float32)
        result = model.generate(input=dummy_audio, batch_size_s=300, disable_pbar=True)
        print(f"Test result: {result}")
        print("Model is working correctly!")

    except Exception as e:
        print(f"ERROR: Failed to fetch model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
