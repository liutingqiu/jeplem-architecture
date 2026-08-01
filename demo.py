"""Run a small interruption-and-recovery PLWM demonstration."""

from pathlib import Path
from tempfile import TemporaryDirectory

from plwm import CognitiveAction, PLWM, PLWMConfig


def main() -> None:
    model = PLWM(PLWMConfig(input_dim=2, latent_dim=8, thought_steps=4))
    model.set_goal([1.0, -1.0], commitment="recover after interruption")
    for observation in ([3.0, -2.0], [3.0, -2.0], None, None):
        result = model.step(observation)
        print(
            f"t={result.time} action={result.action.value} "
            f"error={result.prediction_error:.3f} thoughts={result.thought_steps} "
            f"memory={len(model.state.memory)}"
        )

    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "state.json"
        model.save(str(checkpoint))
        restored = PLWM(model.config)
        restored.load(str(checkpoint))
        result = restored.step(action=CognitiveAction.THINK)
        print(f"restored t={result.time} action={result.action.value} commitment={restored.state.pending_commitment}")


if __name__ == "__main__":
    main()
