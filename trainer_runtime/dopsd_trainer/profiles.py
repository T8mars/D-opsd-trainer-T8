from __future__ import annotations

from copy import deepcopy
from typing import Any


PROFILE_ORDER = (
    "flux2-klein-identity",
    "flux2-klein-editing",
    "z-image-turbo-vlm",
)

PRODUCTION_PROFILES: dict[str, dict[str, Any]] = {
    "flux2-klein-identity": {
        "id": "flux2_identity_recommended_16gb",
        "recipe_id": "flux2-klein-identity",
        "label": "FLUX2 Identity 16GB starter",
        "tier": "recommended_16gb",
        "hardware_profile": "NVIDIA GeForce RTX 4060 Ti 16GB",
        "runner_script": "scripts/run_flux2_smoke.sh",
        "timeout_seconds": 4200,
        "launcher": "python",
        "low_vram": True,
        "use_8bit_adam": True,
        "block_offload": False,
        "block_offload_num_blocks": 2,
        "resolution_scale": "0.625",
        "sample_resolution_scale": "0.5",
        "final_sample_resolution_scale": "0.5",
        "max_train_steps": 5,
        "sample_steps": 5,
        "checkpoint_steps": 5,
        "save_samples": True,
        "save_checkpoints": True,
        "artifact_mode": "sample_and_checkpoint",
        "evidence": [
            "flux2_identity_res0625_artifacts_scale05_5step_20260623045623",
            "flux2_identity_res0625_5step_noartifact_logits1_20260622220405",
        ],
        "cautions": [
            "Native full-size 0.625 inline sample writing OOMed; keep SAMPLE_RESOLUTION_SCALE=0.5.",
            "Artifact writing beyond five steps is not verified on the 16GB profile.",
        ],
    },
    "flux2-klein-editing": {
        "id": "flux2_editing_recommended_16gb",
        "recipe_id": "flux2-klein-editing",
        "label": "FLUX2 Editing 16GB starter",
        "tier": "recommended_16gb",
        "hardware_profile": "NVIDIA GeForce RTX 4060 Ti 16GB",
        "runner_script": "scripts/run_flux2_editing_smoke.sh",
        "timeout_seconds": 4200,
        "launcher": "python",
        "low_vram": True,
        "use_8bit_adam": True,
        "block_offload": False,
        "block_offload_num_blocks": 2,
        "resolution_scale": "0.5625",
        "sample_resolution_scale": "0.5",
        "final_sample_resolution_scale": None,
        "max_train_steps": 5,
        "sample_steps": 5,
        "checkpoint_steps": 5,
        "save_samples": True,
        "save_checkpoints": True,
        "artifact_mode": "sample_and_checkpoint",
        "evidence": [
            "flux2_editing_res05625_artifacts_scale05_5step_20260623044037",
            "flux2_editing_res05625_sample_scale05_1step_20260623034019",
        ],
        "cautions": [
            "Native full-size 0.375 sample writing OOMed; keep SAMPLE_RESOLUTION_SCALE=0.5.",
            "Artifact writing beyond five steps and resolutions above 0.5625 are not verified.",
        ],
    },
    "z-image-turbo-vlm": {
        "id": "zimage_turbo_recommended_16gb",
        "recipe_id": "z-image-turbo-vlm",
        "label": "Z-Image Turbo 16GB starter",
        "tier": "recommended_16gb",
        "hardware_profile": "NVIDIA GeForce RTX 4060 Ti 16GB",
        "runner_script": "scripts/run_zimage_smoke.sh",
        "timeout_seconds": 3600,
        "launcher": "python",
        "low_vram": True,
        "use_8bit_adam": True,
        "block_offload": False,
        "block_offload_num_blocks": 2,
        "resolution_scale": "0.5",
        "sample_resolution_scale": None,
        "final_sample_resolution_scale": None,
        "max_train_steps": 2,
        "sample_steps": 2,
        "checkpoint_steps": 2,
        "save_samples": True,
        "save_checkpoints": True,
        "artifact_mode": "sample_and_checkpoint",
        "evidence": [
            "zimage_style_res05_artifacts_2step_202606221528",
            "zimage_style_res075_artifacts_1step_202606221705",
        ],
        "cautions": [
            "0.75 artifact writing is verified but close to the VRAM ceiling.",
            "1.0 full-resolution no-artifact training OOMs on the 16GB profile.",
        ],
    },
}


def list_production_profiles() -> list[dict[str, Any]]:
    return [get_production_profile(recipe_id) for recipe_id in PROFILE_ORDER]


def get_production_profile(recipe_id: str) -> dict[str, Any]:
    try:
        return deepcopy(PRODUCTION_PROFILES[recipe_id])
    except KeyError as exc:
        raise KeyError(f"unknown production profile recipe: {recipe_id}") from exc


def profile_runner_env(recipe_id: str) -> dict[str, str]:
    profile = get_production_profile(recipe_id)
    env = {
        "RESOLUTION_SCALE": profile["resolution_scale"],
        "MAX_TRAIN_STEPS": str(profile["max_train_steps"]),
        "SAVE_SAMPLES": "1" if profile["save_samples"] else "0",
        "SAVE_CHECKPOINTS": "1" if profile["save_checkpoints"] else "0",
        "SAMPLE_STEPS": str(profile["sample_steps"]),
        "CHECKPOINT_STEPS": str(profile["checkpoint_steps"]),
        "BLOCK_OFFLOAD": "1" if profile["block_offload"] else "0",
    }
    if profile["sample_resolution_scale"] is not None:
        env["SAMPLE_RESOLUTION_SCALE"] = str(profile["sample_resolution_scale"])
    if profile["final_sample_resolution_scale"] is not None:
        env["FINAL_SAMPLE_RESOLUTION_SCALE"] = str(profile["final_sample_resolution_scale"])
    return env


def build_profile_runner_command(recipe_id: str, exp_name: str) -> str:
    profile = get_production_profile(recipe_id)
    env = {"EXP_NAME": exp_name, **profile_runner_env(recipe_id)}
    env_prefix = " ".join(f"{key}={value}" for key, value in env.items())
    return f"{env_prefix} timeout {profile['timeout_seconds']} bash {profile['runner_script']}"
