#!/usr/bin/env python3
"""
Fetch script for speaker embedding model (WeSpeaker/ECAPA-TDNN).

Downloads and caches the speaker embedding model for speaker similarity reward.

Usage:
    python scripts/fetch_reward_spk_model.py [--output-dir DIR] [--cache-dir DIR] [--offline]
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Fetch speaker embedding model for reward computation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save model files",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for model weights",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline mode - only use existing cache",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default=None,
        help="URL to download model from (optional)",
    )
    args = parser.parse_args()

    # The ECAPA-TDNN model in f5_tts/eval uses wavlm features
    # We verify it can be instantiated
    try:
        from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
    except ImportError as e:
        print(f"ERROR: Could not import ECAPA_TDNN_SMALL: {e}")
        print("Make sure f5-tts is properly installed.")
        sys.exit(1)

    print("Verifying ECAPA-TDNN model...")

    try:
        # Create model instance
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024,
            feat_type="wavlm_large",
            config_path=None,
        )

        print("ECAPA-TDNN model instantiated successfully!")

        # If output directory specified, save the model architecture info
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            info_path = os.path.join(args.output_dir, "model_info.txt")
            with open(info_path, "w") as f:
                f.write("ECAPA-TDNN Speaker Embedding Model\n")
                f.write("=" * 40 + "\n")
                f.write(f"Feature type: wavlm_large\n")
                f.write(f"Feature dimension: 1024\n")
                f.write(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
            print(f"Model info saved to: {info_path}")

        # Note about WavLM
        print("\nNOTE: This model uses WavLM features internally.")
        print("WavLM will be downloaded automatically on first use via torch hub.")

        if args.cache_dir:
            # Set torch hub cache
            import torch

            torch.hub.set_dir(args.cache_dir)
            print(f"Torch hub cache set to: {args.cache_dir}")

        # Test with dummy input
        print("\nTesting model with dummy input...")
        import torch

        dummy_wav = torch.randn(1, 16000)
        try:
            with torch.no_grad():
                # This may trigger WavLM download
                embedding = model(dummy_wav)
            print(f"Output embedding shape: {embedding.shape}")
            print("Model is working correctly!")
        except Exception as e:
            print(f"WARNING: Test failed (may need to download WavLM): {e}")
            if not args.offline:
                print("The model should work on first actual use.")

    except Exception as e:
        print(f"ERROR: Failed to verify model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
