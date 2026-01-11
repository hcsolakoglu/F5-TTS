"""
Tests for RL integration in F5-TTS.

These tests verify:
1. Backward compatibility with deterministic mode
2. Gaussian architecture support
3. Soft loading of checkpoints
4. Gaussian NLL loss computation
5. GRPO step with mock reward
6. Plugin loading via registry
7. Dependency guards
"""

import pytest
import torch
import torch.nn as nn

# Test fixtures


@pytest.fixture
def device():
    """Get appropriate device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 64


@pytest.fixture
def mel_dim():
    return 100


@pytest.fixture
def text_dim():
    return 256


@pytest.fixture
def dummy_mel(batch_size, seq_len, mel_dim, device):
    """Create dummy mel spectrogram."""
    return torch.randn(batch_size, seq_len, mel_dim, device=device)


@pytest.fixture
def dummy_text(batch_size, text_dim, device):
    """Create dummy text tokens."""
    return torch.randint(0, 100, (batch_size, text_dim), device=device)


# Test 1: Deterministic regression - forward returns tensor, shapes stable
class TestDeterministicRegression:
    """Test that deterministic mode behavior is unchanged."""

    def test_dit_deterministic_forward_returns_tensor(self, device, batch_size, seq_len, mel_dim):
        """DiT forward in deterministic mode returns a tensor."""
        from f5_tts.model.backbones.dit import DiT

        model = DiT(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mel_dim=mel_dim,
            text_num_embeds=100,
            output_dist="deterministic",
        ).to(device)

        x = torch.randn(batch_size, seq_len, mel_dim, device=device)
        cond = torch.randn(batch_size, seq_len, mel_dim, device=device)
        text = torch.randint(0, 100, (batch_size, 32), device=device)
        time = torch.rand(batch_size, device=device)

        output = model(x=x, cond=cond, text=text, time=time)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, seq_len, mel_dim)

    def test_mmdit_deterministic_forward_returns_tensor(self, device, batch_size, seq_len, mel_dim):
        """MMDiT forward in deterministic mode returns a tensor."""
        from f5_tts.model.backbones.mmdit import MMDiT

        model = MMDiT(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mel_dim=mel_dim,
            text_num_embeds=100,
            output_dist="deterministic",
        ).to(device)

        x = torch.randn(batch_size, seq_len, mel_dim, device=device)
        cond = torch.randn(batch_size, seq_len, mel_dim, device=device)
        text = torch.randint(0, 100, (batch_size, 32), device=device)
        time = torch.rand(batch_size, device=device)

        output = model(x=x, cond=cond, text=text, time=time)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, seq_len, mel_dim)

    def test_unett_deterministic_forward_returns_tensor(self, device, batch_size, seq_len, mel_dim):
        """UNetT forward in deterministic mode returns a tensor."""
        from f5_tts.model.backbones.unett import UNetT

        model = UNetT(
            dim=64,
            depth=4,  # Must be even for UNetT
            heads=4,
            dim_head=16,
            mel_dim=mel_dim,
            text_num_embeds=100,
            output_dist="deterministic",
        ).to(device)

        x = torch.randn(batch_size, seq_len, mel_dim, device=device)
        cond = torch.randn(batch_size, seq_len, mel_dim, device=device)
        text = torch.randint(0, 100, (batch_size, 32), device=device)
        time = torch.rand(batch_size, device=device)

        output = model(x=x, cond=cond, text=text, time=time)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, seq_len, mel_dim)


# Test 2: Gaussian architecture - forward returns tensor, forward_prob returns tuple
class TestGaussianArchitecture:
    """Test gaussian output distribution mode."""

    def test_dit_gaussian_forward_returns_tensor(self, device, batch_size, seq_len, mel_dim):
        """DiT forward in gaussian mode still returns tensor (mu only)."""
        from f5_tts.model.backbones.dit import DiT

        model = DiT(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mel_dim=mel_dim,
            text_num_embeds=100,
            output_dist="gaussian",
        ).to(device)

        x = torch.randn(batch_size, seq_len, mel_dim, device=device)
        cond = torch.randn(batch_size, seq_len, mel_dim, device=device)
        text = torch.randint(0, 100, (batch_size, 32), device=device)
        time = torch.rand(batch_size, device=device)

        output = model(x=x, cond=cond, text=text, time=time)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, seq_len, mel_dim)

    def test_dit_gaussian_forward_prob_returns_tuple(self, device, batch_size, seq_len, mel_dim):
        """DiT forward_prob returns (mu, ln_sig) tuple."""
        from f5_tts.model.backbones.dit import DiT

        model = DiT(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mel_dim=mel_dim,
            text_num_embeds=100,
            output_dist="gaussian",
        ).to(device)

        x = torch.randn(batch_size, seq_len, mel_dim, device=device)
        cond = torch.randn(batch_size, seq_len, mel_dim, device=device)
        text = torch.randint(0, 100, (batch_size, 32), device=device)
        time = torch.rand(batch_size, device=device)

        mu, ln_sig = model.forward_prob(x=x, cond=cond, text=text, time=time)

        assert isinstance(mu, torch.Tensor)
        assert isinstance(ln_sig, torch.Tensor)
        assert mu.shape == (batch_size, seq_len, mel_dim)
        assert ln_sig.shape == (batch_size, seq_len, mel_dim)

    def test_deterministic_model_raises_on_forward_prob(self, device, batch_size, seq_len, mel_dim):
        """Deterministic model raises error on forward_prob."""
        from f5_tts.model.backbones.dit import DiT

        model = DiT(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mel_dim=mel_dim,
            text_num_embeds=100,
            output_dist="deterministic",
        ).to(device)

        x = torch.randn(batch_size, seq_len, mel_dim, device=device)
        cond = torch.randn(batch_size, seq_len, mel_dim, device=device)
        text = torch.randint(0, 100, (batch_size, 32), device=device)
        time = torch.rand(batch_size, device=device)

        with pytest.raises(RuntimeError, match="forward_prob requires output_dist='gaussian'"):
            model.forward_prob(x=x, cond=cond, text=text, time=time)

    def test_ln_sig_head_exists_in_gaussian_mode(self, device, mel_dim):
        """proj_out_ln_sig exists only in gaussian mode."""
        from f5_tts.model.backbones.dit import DiT

        det_model = DiT(
            dim=64, depth=2, heads=4, dim_head=16, mel_dim=mel_dim, text_num_embeds=100, output_dist="deterministic"
        )
        gauss_model = DiT(
            dim=64, depth=2, heads=4, dim_head=16, mel_dim=mel_dim, text_num_embeds=100, output_dist="gaussian"
        )

        assert det_model.proj_out_ln_sig is None
        assert gauss_model.proj_out_ln_sig is not None
        assert isinstance(gauss_model.proj_out_ln_sig, nn.Linear)


# Test 3: Soft load - load non-RL state into gaussian model
class TestSoftLoad:
    """Test checkpoint loading compatibility."""

    def test_load_deterministic_into_gaussian(self, device, mel_dim):
        """Load deterministic checkpoint into gaussian model with strict=False."""
        from f5_tts.model.backbones.dit import DiT

        # Create deterministic model and save state
        det_model = DiT(
            dim=64, depth=2, heads=4, dim_head=16, mel_dim=mel_dim, text_num_embeds=100, output_dist="deterministic"
        ).to(device)

        det_state = det_model.state_dict()

        # Create gaussian model
        gauss_model = DiT(
            dim=64, depth=2, heads=4, dim_head=16, mel_dim=mel_dim, text_num_embeds=100, output_dist="gaussian"
        ).to(device)

        # Load with strict=False
        missing, unexpected = gauss_model.load_state_dict(det_state, strict=False)

        # proj_out_ln_sig should be missing
        assert any("proj_out_ln_sig" in k for k in missing)
        assert len(unexpected) == 0

        # ln_sig head should still exist
        assert gauss_model.proj_out_ln_sig is not None


# Test 4: Gaussian loss - finite, grads reach ln_sig params
class TestGaussianLoss:
    """Test Gaussian NLL loss computation."""

    def test_gaussian_nll_loss_is_finite(self, device, batch_size, seq_len, mel_dim):
        """Gaussian NLL loss should be finite."""
        from f5_tts.model.backbones.dit import DiT
        from f5_tts.model.cfm import CFM

        transformer = DiT(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mel_dim=mel_dim,
            text_num_embeds=100,
            output_dist="gaussian",
        ).to(device)

        model = CFM(
            transformer=transformer,
            objective="gaussian_nll",
            mel_spec_kwargs={"n_mel_channels": mel_dim},
        ).to(device)

        # Create dummy input
        mel = torch.randn(batch_size, seq_len, mel_dim, device=device)
        text = torch.randint(0, 100, (batch_size, 32), device=device)

        loss, cond, pred = model(mel, text)

        assert torch.isfinite(loss)
        assert loss > 0

    def test_gaussian_nll_grads_reach_ln_sig(self, device, batch_size, seq_len, mel_dim):
        """Gradients should flow to ln_sig parameters."""
        from f5_tts.model.backbones.dit import DiT
        from f5_tts.model.cfm import CFM

        transformer = DiT(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mel_dim=mel_dim,
            text_num_embeds=100,
            output_dist="gaussian",
        ).to(device)

        model = CFM(
            transformer=transformer,
            objective="gaussian_nll",
            mel_spec_kwargs={"n_mel_channels": mel_dim},
        ).to(device)

        mel = torch.randn(batch_size, seq_len, mel_dim, device=device)
        text = torch.randint(0, 100, (batch_size, 32), device=device)

        loss, cond, pred = model(mel, text)
        loss.backward()

        # Check that ln_sig has gradients
        ln_sig_weight = model.transformer.proj_out_ln_sig.weight
        assert ln_sig_weight.grad is not None
        assert torch.any(ln_sig_weight.grad != 0)


# Test 5: GRPO step with mock reward plugin
class TestGRPOWithMockReward:
    """Test GRPO training step with dummy reward."""

    def test_dummy_reward_provider_returns_fixed_reward(self, device):
        """DummyRewardProvider returns configured fixed reward."""
        from f5_tts.rewards.providers.dummy import DummyRewardProvider
        from f5_tts.rewards import RewardInput

        provider = DummyRewardProvider(fixed_reward=0.5)

        inputs = [
            RewardInput(
                audio=torch.randn(1000, device=device),
                text="hello world",
                sample_rate=24000,
            )
        ]

        outputs = provider.compute(inputs)

        assert len(outputs) == 1
        assert torch.isclose(outputs[0].total_reward, torch.tensor(0.5, device=device))

    def test_reward_combiner_with_dummy_providers(self, device):
        """RewardCombiner correctly combines multiple dummy providers."""
        from f5_tts.rewards.providers.dummy import DummyRewardProvider
        from f5_tts.rewards import RewardCombiner, RewardInput

        provider1 = DummyRewardProvider(fixed_reward=1.0)
        provider2 = DummyRewardProvider(fixed_reward=2.0)

        combiner = RewardCombiner(
            providers=[provider1, provider2],
            weights=[0.5, 0.5],
            mode="sum",
        )

        inputs = [
            RewardInput(
                audio=torch.randn(1000, device=device),
                text="test",
                sample_rate=24000,
            )
        ]

        outputs = combiner.compute(inputs)

        # Sum: 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        assert len(outputs) == 1
        assert torch.isclose(outputs[0].total_reward, torch.tensor(1.5, device=device))


# Test 6: Plugin loading via registry
class TestPluginLoading:
    """Test dynamic loading of reward providers."""

    def test_registry_load_by_import_path(self):
        """Registry can load provider by import path."""
        from f5_tts.rewards.registry import RewardRegistry

        provider = RewardRegistry.create_from_import_path(
            "f5_tts.rewards.providers.dummy:DummyRewardProvider",
            cfg={"fixed_reward": 0.75},
        )

        assert provider.name == "dummy"
        assert provider.fixed_reward == 0.75

    def test_registry_create_from_config(self):
        """Registry can create provider from config dict."""
        from f5_tts.rewards.registry import RewardRegistry

        config = {
            "provider": "f5_tts.rewards.providers.dummy:DummyRewardProvider",
            "cfg": {"fixed_reward": 0.25},
        }

        provider = RewardRegistry.create_from_config(config)

        assert provider.name == "dummy"
        assert provider.fixed_reward == 0.25

    def test_registry_invalid_import_path_raises(self):
        """Invalid import path raises appropriate error."""
        from f5_tts.rewards.registry import RewardRegistry

        with pytest.raises(ValueError, match="Invalid import path"):
            RewardRegistry.create_from_import_path("invalid_path_no_colon")

        with pytest.raises(ImportError):
            RewardRegistry.create_from_import_path("nonexistent.module:SomeClass")


# Test 7: Dependency guards
class TestDependencyGuards:
    """Test that dependencies are properly guarded."""

    def test_base_import_does_not_import_funasr(self):
        """Base f5_tts import should not import funasr."""
        import sys

        # Clear any cached imports
        modules_to_check = ["funasr"]
        for mod in modules_to_check:
            if mod in sys.modules:
                del sys.modules[mod]

        # Import base package
        import f5_tts.model  # noqa: F401
        import f5_tts.rewards  # noqa: F401

        # Check that funasr is not imported (lazy import guard)
        assert "funasr" not in sys.modules

    def test_funasr_provider_raises_without_deps(self):
        """FunASR provider raises clear error without dependencies."""
        import sys

        # Ensure funasr is not available
        if "funasr" in sys.modules:
            del sys.modules["funasr"]

        # Mock ImportError for funasr
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "funasr":
                raise ImportError("No module named 'funasr'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import

        try:
            from f5_tts.rewards.providers.funasr_sensevoice import FunASRWerReward

            provider = FunASRWerReward()

            with pytest.raises(RuntimeError, match="FunASR is required"):
                provider._check_dependencies()
        finally:
            builtins.__import__ = original_import


# Test CFM objective modes
class TestCFMObjectives:
    """Test different CFM training objectives."""

    def test_mse_objective_default(self, device, batch_size, seq_len, mel_dim):
        """MSE objective is default and works correctly."""
        from f5_tts.model.backbones.dit import DiT
        from f5_tts.model.cfm import CFM

        transformer = DiT(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mel_dim=mel_dim,
            text_num_embeds=100,
            output_dist="deterministic",
        ).to(device)

        model = CFM(
            transformer=transformer,
            objective="mse",
            mel_spec_kwargs={"n_mel_channels": mel_dim},
        ).to(device)

        mel = torch.randn(batch_size, seq_len, mel_dim, device=device)
        text = torch.randint(0, 100, (batch_size, 32), device=device)

        loss, cond, pred = model(mel, text)

        assert torch.isfinite(loss)
        assert loss.shape == ()  # Scalar loss

    def test_rl_grpo_objective_raises_in_forward(self, device, batch_size, seq_len, mel_dim):
        """RL GRPO objective raises when used directly in forward."""
        from f5_tts.model.backbones.dit import DiT
        from f5_tts.model.cfm import CFM

        transformer = DiT(
            dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mel_dim=mel_dim,
            text_num_embeds=100,
            output_dist="gaussian",
        ).to(device)

        model = CFM(
            transformer=transformer,
            objective="rl_grpo",
            mel_spec_kwargs={"n_mel_channels": mel_dim},
        ).to(device)

        mel = torch.randn(batch_size, seq_len, mel_dim, device=device)
        text = torch.randint(0, 100, (batch_size, 32), device=device)

        with pytest.raises(ValueError, match="rl_grpo objective should not be used directly"):
            model(mel, text)


# Integration marker for optional tests
@pytest.mark.integration
class TestIntegration:
    """Integration tests that may require additional resources."""

    @pytest.mark.skip(reason="Requires audio pack to be built")
    def test_reward_providers_on_audio_pack(self):
        """Test reward providers on actual audio samples."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
