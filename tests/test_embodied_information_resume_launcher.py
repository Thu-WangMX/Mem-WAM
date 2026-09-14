from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "ops" / "resume_putback_embodied_information_10k_to40k.sh"


def test_resume_launcher_uses_two_tier_checkpoint_schedule():
    text = LAUNCHER.read_text()

    assert "step_010000" in text
    assert "step_015000" in text
    assert "save_every=5000" in text
    assert "save_training_state_every=10000" in text
    assert "keep_training_state_checkpoints=1" in text
    assert "/mnt/vepfs01/output/kevin.wang/memorywam/train/" in text
    assert "/mnt/vepfs02/output/kevin.wang/memorywam/" in text
    assert "/output/kevin_wang/" not in text
    assert "/mnt/vepfs02/output/kevin.wang/memorywam/model_assets/diffsynth_official" in text
    assert "/mnt/vepfs02/datasets/kevin_wang" not in text
    assert "step_015000.pt" not in text
