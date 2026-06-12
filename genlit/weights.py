"""Checkpoint resolution: local path or HuggingFace Hub auto-download."""
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


_MODE_TO_SUBFOLDER = {
    "single": "single_object",
    "mit": "mit",
    "multi": "multi_object",
}


def resolve_checkpoint(
    mode: str,
    override: Optional[str] = None,
    repo_id: str = "sbharadwaj/genlit",
) -> str:
    """Return path to the ControlNet checkpoint dir for the given mode.

    Args:
        mode: one of 'single', 'mit', 'multi'
        override: explicit path to checkpoint dir. If set, returned as-is.
        repo_id: HF Hub repo id (wired up at M6 with the actual user name).

    Returns:
        Path string suitable for ControlNetSDVModel.from_pretrained(path, subfolder="controlnet").
    """
    if override is not None:
        return str(override)

    if mode not in _MODE_TO_SUBFOLDER:
        raise ValueError(
            f"Unknown mode {mode!r}; expected one of {list(_MODE_TO_SUBFOLDER)}"
        )

    subfolder = _MODE_TO_SUBFOLDER[mode]
    local_root = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subfolder}/**"],
    )
    return str(Path(local_root) / subfolder / "controlnet")
