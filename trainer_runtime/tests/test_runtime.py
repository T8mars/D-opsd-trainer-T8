from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "trainer_runtime"))

from dopsd_trainer.datasets import validate_dataset
from dopsd_trainer.models import inspect_model_cache
from dopsd_trainer.outputs import parse_loss_jsonl, summarize_run
from dopsd_trainer.recipes import TrainingConfig, build_accelerate_command
from dopsd_trainer.settings import build_settings_summary


class RuntimeTests(unittest.TestCase):
    def test_validate_bundled_zimage_dataset(self) -> None:
        summary = validate_dataset(
            "z-image-turbo_self-distill-vlm/dataset/style_Millennium/data.jsonl",
            "z-image-turbo-vlm",
            PROJECT_ROOT,
        )
        self.assertTrue(summary.ok)
        self.assertEqual(summary.rows, 4)
        self.assertEqual(summary.valid_rows, 4)

    def test_dataset_summary_reports_bucket_dimensions(self) -> None:
        summary = validate_dataset(
            "flux2-klein_self-distill-edit/dataset/corgi/data.jsonl",
            "flux2-klein-identity",
            PROJECT_ROOT,
        ).to_dict()
        self.assertEqual(
            summary["bucket_summary"],
            [
                {"width": 780, "height": 780, "count": 3},
                {"width": 781, "height": 781, "count": 1},
            ],
        )

    def test_validate_bundled_edit_dataset_requires_pairs(self) -> None:
        summary = validate_dataset(
            "flux2-klein-edit-self-distill-gt-ref/dataset/interaction/data.jsonl",
            "flux2-klein-editing",
            PROJECT_ROOT,
        )
        self.assertTrue(summary.ok)
        self.assertEqual(summary.rows, 16)

    def test_dataset_manager_supports_ai_toolkit_style_upload_and_item_editing(self) -> None:
        datasets_lib_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "datasets.ts"
        datasets_route_path = PROJECT_ROOT / "trainer-ui" / "src" / "app" / "api" / "datasets" / "route.ts"
        dataset_ui_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "DatasetValidator.tsx"
        i18n_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx"

        datasets_lib_source = datasets_lib_path.read_text(encoding="utf-8")
        datasets_route_source = datasets_route_path.read_text(encoding="utf-8")
        dataset_ui_source = dataset_ui_path.read_text(encoding="utf-8")
        i18n_source = i18n_path.read_text(encoding="utf-8")

        self.assertIn("managed-datasets.json", datasets_lib_source)
        self.assertIn("readManagedDatasets", datasets_lib_source)
        self.assertIn("importManagedDataset", datasets_lib_source)
        self.assertIn("addManagedDatasetItem", datasets_lib_source)
        self.assertIn("updateManagedDatasetItem", datasets_lib_source)
        self.assertIn("deleteManagedDatasetItem", datasets_lib_source)
        self.assertIn("deleteManagedDataset", datasets_lib_source)
        self.assertIn("combineDatasetSelections", datasets_lib_source)
        self.assertIn("captionFileByStem", datasets_lib_source)
        self.assertIn("local_path_list", datasets_lib_source)
        self.assertIn("user_prompt_en", datasets_lib_source)

        self.assertIn("request.formData", datasets_route_source)
        self.assertIn("import-files", datasets_route_source)
        self.assertIn("add-item", datasets_route_source)
        self.assertIn("update-item", datasets_route_source)
        self.assertIn("delete-item", datasets_route_source)
        self.assertIn("delete-dataset", datasets_route_source)
        self.assertIn("combine-selection", datasets_route_source)

        self.assertIn('type="file"', dataset_ui_source)
        self.assertIn("multiple", dataset_ui_source)
        self.assertIn("selectedDatasetPaths", dataset_ui_source)
        self.assertIn("handleImportDataset", dataset_ui_source)
        self.assertIn("handleAddDatasetItem", dataset_ui_source)
        self.assertIn("handleUpdateDatasetItem", dataset_ui_source)
        self.assertIn("handleDeleteDatasetItem", dataset_ui_source)
        self.assertIn("handleDeleteDataset", dataset_ui_source)

        for text in (
            "导入数据集",
            "上传图像",
            "打标文件",
            "多选数据集",
            "编辑打标",
            "删除图片",
            "新增图片",
        ):
            self.assertIn(text, i18n_source)

    def test_new_job_accepts_multiple_dataset_paths_and_combines_them_before_draft(self) -> None:
        wizard_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "NewJobWizard.tsx"
        jobs_route_path = PROJECT_ROOT / "trainer-ui" / "src" / "app" / "api" / "jobs" / "route.ts"
        jobs_lib_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        i18n_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx"

        wizard_source = wizard_path.read_text(encoding="utf-8")
        jobs_route_source = jobs_route_path.read_text(encoding="utf-8")
        jobs_lib_source = jobs_lib_path.read_text(encoding="utf-8")
        i18n_source = i18n_path.read_text(encoding="utf-8")

        self.assertIn("selectedDatasetPaths", wizard_source)
        self.assertIn("toggleDatasetPath", wizard_source)
        self.assertIn("datasetPaths", wizard_source)
        self.assertIn("selectedDatasets", wizard_source)
        self.assertIn("multiDatasetSelection", wizard_source)
        self.assertIn('type="checkbox"', wizard_source)

        self.assertIn("datasetPaths?: string[]", jobs_route_source)
        self.assertIn("createDraftJob(payload.recipeId, payload.datasetPath, payload.datasetPaths, payload.trainingOverrides)", jobs_route_source)

        self.assertIn("datasetPaths?: string[]", jobs_lib_source)
        self.assertIn("preflightDraftDatasetPaths", jobs_lib_source)
        self.assertIn("combineDatasetSelections", jobs_lib_source)
        self.assertIn("Combined dataset selection", jobs_lib_source)
        self.assertIn("datasetPaths:", jobs_lib_source)

        self.assertIn("多选数据集", i18n_source)
        self.assertIn("已选择数据集", i18n_source)
        self.assertIn("合并后创建草稿", i18n_source)

    def test_runner_commands_pass_selected_dataset_jsonl_to_training_scripts(self) -> None:
        jobs_lib_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        jobs_lib_source = jobs_lib_path.read_text(encoding="utf-8")

        self.assertIn("DATA_PATH_TRAIN_JSONL", jobs_lib_source)
        self.assertIn("DATA_PATH_TEST_JSONL", jobs_lib_source)
        self.assertIn("toWslPath(resolveProjectPath(job.datasetPath))", jobs_lib_source)

        for relative in (
            "scripts/run_flux2_smoke.sh",
            "scripts/run_flux2_editing_smoke.sh",
            "scripts/run_zimage_smoke.sh",
        ):
            with self.subTest(relative=relative):
                source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
                self.assertIn("DATA_PATH_TRAIN_JSONL", source)
                self.assertIn("DATA_PATH_TEST_JSONL", source)
                self.assertIn('--data-path-train-jsonl "$DATA_PATH_TRAIN_JSONL"', source)
                self.assertIn('--data-path-test-jsonl "$DATA_PATH_TEST_JSONL"', source)

    def test_new_job_ui_sends_ai_toolkit_style_training_controls(self) -> None:
        wizard_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "NewJobWizard.tsx"
        jobs_route_path = PROJECT_ROOT / "trainer-ui" / "src" / "app" / "api" / "jobs" / "route.ts"
        i18n_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx"

        wizard_source = wizard_path.read_text(encoding="utf-8")
        jobs_route_source = jobs_route_path.read_text(encoding="utf-8")
        i18n_source = i18n_path.read_text(encoding="utf-8")

        for token in (
            "trainingOverrides",
            "maxTrainSteps",
            "learningRateGen",
            "gradientAccumulationSteps",
            "batchSize",
            "checkpointSteps",
            "sampleSteps",
            "targetResolution",
            "skipInitialSample",
            "saveSamples",
            "saveCheckpoints",
        ):
            self.assertIn(token, wizard_source)

        self.assertIn("trainingOverrides?: TrainingOverrides", jobs_route_source)
        self.assertIn("payload.trainingOverrides", jobs_route_source)

        for text in (
            "训练参数",
            "总步数",
            "学习率",
            "梯度累积",
            "批大小",
            "保存间隔",
            "采样间隔",
            "目标分辨率",
            "跳过首采样",
        ):
            self.assertIn(text, i18n_source)

        for forbidden in (
            "| 'epochs'",
            "epochs: 2",
            "`EPOCHS=${trainingOverrides.epochs}`",
            "['epochs', t('epochs')",
        ):
            self.assertNotIn(forbidden, wizard_source)
        self.assertNotIn("训练轮数", i18n_source)

    def test_job_training_overrides_flow_into_runner_environment(self) -> None:
        jobs_lib_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        jobs_lib_source = jobs_lib_path.read_text(encoding="utf-8")

        self.assertIn("export type TrainingOverrides", jobs_lib_source)
        self.assertIn("trainingOverrides?: TrainingOverrides", jobs_lib_source)
        self.assertIn("createDraftJob(", jobs_lib_source)
        self.assertIn("normalizeTrainingOverrides", jobs_lib_source)
        self.assertIn("profileEnvAssignments(profile, job.expName", jobs_lib_source)
        self.assertIn("timeoutForJob(profile, job)", jobs_lib_source)
        self.assertIn("internalEpochsForMaxTrainSteps", jobs_lib_source)
        self.assertNotIn("epochs: overrides?.epochs ?? 2", jobs_lib_source)

        for env_var in (
            "EPOCHS",
            "LEARNING_RATE_GEN",
            "BATCH_SIZE",
            "GRADIENT_ACCUMULATION_STEPS",
            "TARGET_RESOLUTION",
            "SKIP_INITIAL_SAMPLE",
            "SAVE_SAMPLES",
            "SAVE_CHECKPOINTS",
            "USE_8BIT_ADAM",
            "LOW_VRAM",
            "BLOCK_OFFLOAD_NUM_BLOCKS",
        ):
            self.assertIn(f"{env_var}=", jobs_lib_source)

    def test_training_config_v2_schema_migration_and_bounds(self) -> None:
        config_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "trainingConfig.ts"
        source = config_path.read_text(encoding="utf-8")

        for token in (
            "export type TrainingOverridesV2",
            "basics:",
            "lora:",
            "optimizer:",
            "sampling:",
            "datasets:",
            "memory:",
            "advancedDopsd:",
            "migrateTrainingOverridesToV2",
            "normalizeTrainingConfigV2",
            "version: 2",
        ):
            self.assertIn(token, source)

        for bounded_token in (
            "maxTrainSteps: boundedInt(legacy.maxTrainSteps, 1, 200000)",
            "sampleSteps: boundedInt(legacy.sampleSteps, 0, 200000)",
            "learningRateGen: boundedFloat(legacy.learningRateGen, 1e-8, 1)",
            "captionDropout: boundedFloat(legacy.captionDropout, 0, 1)",
            "weight: boundedFloat(dataset.weight, 0, 10)",
            "rank: boundedInt(legacy.networkDim ?? legacy.rank, 1, 512)",
            "tileSize: boundedInt(legacy.tileSize, 16, 512)",
        ):
            self.assertIn(bounded_token, source)

    def test_jobs_store_v2_training_config_and_build_runner_from_it(self) -> None:
        jobs_lib_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        jobs_lib_source = jobs_lib_path.read_text(encoding="utf-8")

        for token in (
            "@/lib/trainingConfig",
            "trainingConfig?: TrainingOverridesV2",
            "migrateTrainingOverridesToV2(trainingOverrides",
            "trainingConfig,",
            "defaultTrainingValues(base.id, profile, trainingConfig)",
            "commandForProductionProfile(base.id, expName, datasetPreflight?.datasetPath, trainingConfig, sampleDatasetPath)",
            "trainingConfigForJob(job)",
            "profileEnvAssignments(profile, job.expName",
        ):
            self.assertIn(token, jobs_lib_source)

    def test_sampling_config_accepts_custom_sample_prompt_table(self) -> None:
        config_source = (PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "trainingConfig.ts").read_text(encoding="utf-8")
        wizard_source = (PROJECT_ROOT / "trainer-ui" / "src" / "components" / "NewJobWizard.tsx").read_text(encoding="utf-8")
        i18n_source = (PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx").read_text(encoding="utf-8")

        for token in (
            "samplePrompts?: string[]",
            "normalizeSamplePrompts",
            "samplePrompts: normalizeSamplePrompts(sampling.samplePrompts)",
            "samplePrompts: normalizeSamplePrompts(legacy.samplePrompts)",
        ):
            self.assertIn(token, config_source)

        for token in (
            "samplePromptText",
            "samplePrompts",
            "customSamplePrompts",
            "样图提示词",
            "一行一个",
        ):
            self.assertIn(token, wizard_source + i18n_source)

    def test_runner_writes_custom_sample_jsonl_for_test_dataset(self) -> None:
        jobs_lib_source = (PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts").read_text(encoding="utf-8")

        for token in (
            "sampleDatasetPath?: string",
            "writeCustomSampleJsonl",
            "sampleJsonlPath",
            "samplePrompts",
            "DATA_PATH_TEST_JSONL=${bashQuote(toWslPath(resolveProjectPath(sampleDatasetPath ?? datasetPath)))}",
            "await writeCustomSampleJsonl(",
            "sampleDatasetPath,",
        ):
            self.assertIn(token, jobs_lib_source)

    def test_dataset_weights_flow_from_new_job_ui_into_combined_jsonl(self) -> None:
        datasets_source = (PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "datasets.ts").read_text(encoding="utf-8")
        jobs_source = (PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts").read_text(encoding="utf-8")
        wizard_source = (PROJECT_ROOT / "trainer-ui" / "src" / "components" / "NewJobWizard.tsx").read_text(encoding="utf-8")
        i18n_source = (PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx").read_text(encoding="utf-8")

        for token in (
            "export type DatasetSelectionInput",
            "normalizeDatasetSelectionInputs",
            "weight: boundedDatasetWeight",
            "for (let copyIndex = 0; copyIndex < weight; copyIndex += 1)",
            "combineDatasetSelections(datasetSelections: DatasetSelectionInput[]",
        ):
            self.assertIn(token, datasets_source)

        for token in (
            "datasetSelectionInputs",
            "trainingConfig.datasets.items",
            "hasWeightedSelection",
            "selectedPaths.length <= 1 && !hasWeightedSelection",
            "combineDatasetSelections(datasetSelectionInputs, recipeId)",
        ):
            self.assertIn(token, jobs_source)

        for token in (
            "datasetWeights",
            "updateDatasetWeight",
            "datasetWeight",
            "t('multiDatasetHelp')",
            "datasets: datasetPaths.map",
            "训练权重",
        ):
            self.assertIn(token, wizard_source + i18n_source)

    def test_smoke_scripts_expose_real_training_controls(self) -> None:
        expected_env = (
            "LEARNING_RATE_GEN",
            "BATCH_SIZE",
            "GRADIENT_ACCUMULATION_STEPS",
            "TARGET_RESOLUTION",
            "SKIP_INITIAL_SAMPLE",
            "USE_8BIT_ADAM",
            "LOW_VRAM",
        )
        expected_args = (
            '--epochs "$EPOCHS"',
            '--learning-rate-gen "$LEARNING_RATE_GEN"',
            '--batch-size "$BATCH_SIZE"',
            '--gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"',
            '--target-resolution "$TARGET_RESOLUTION"',
        )

        for relative in (
            "scripts/run_flux2_smoke.sh",
            "scripts/run_flux2_editing_smoke.sh",
            "scripts/run_zimage_smoke.sh",
        ):
            with self.subTest(relative=relative):
                source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
                self.assertIn('EPOCHS="${EPOCHS:-$((MAX_TRAIN_STEPS + 1))}"', source)
                for token in expected_env:
                    self.assertIn(token, source)
                for token in expected_args:
                    self.assertIn(token, source)
                self.assertIn("TRAIN_ARGS+=(--skip-initial-sample)", source)
                self.assertIn("TRAIN_ARGS+=(--use-8bit-adam)", source)
                self.assertIn("TRAIN_ARGS+=(--low-vram)", source)

    def test_training_scripts_support_target_resolution_skip_first_sample_and_memory_flush(self) -> None:
        for relative in (
            "flux2-klein_self-distill-edit",
            "flux2-klein-edit-self-distill-gt-ref",
            "z-image-turbo_self-distill-vlm",
        ):
            with self.subTest(relative=relative):
                args_source = (PROJECT_ROOT / relative / "arguments.py").read_text(encoding="utf-8")
                train_source = (PROJECT_ROOT / relative / "train_dopsd.py").read_text(encoding="utf-8")

                self.assertIn("--target-resolution", args_source)
                self.assertIn("--skip-initial-sample", args_source)
                self.assertIn("target_resolution", train_source)
                self.assertIn("if not args.skip_initial_sample", train_source)
                self.assertIn("Skipped initial sample generation", train_source)
                self.assertNotIn("if global_step > 1000", train_source)
                self.assertIn("args.sample_steps > 0", train_source)
                self.assertIn("free_cuda_memory()", train_source)

    def test_jobs_page_makes_loss_history_readable(self) -> None:
        jobs_table_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "JobsTable.tsx"
        i18n_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx"

        jobs_table_source = jobs_table_path.read_text(encoding="utf-8")
        i18n_source = i18n_path.read_text(encoding="utf-8")

        self.assertIn("LossHistorySummary", jobs_table_source)
        self.assertIn("lossHistory.slice(-5)", jobs_table_source)
        self.assertIn("lossDopsd", jobs_table_source)
        self.assertIn("gradNorm", jobs_table_source)
        self.assertIn("最近损失", i18n_source)
        self.assertIn("梯度范数", i18n_source)

    def test_detached_wsl_runner_does_not_wait_on_empty_pid(self) -> None:
        jobs_lib_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        jobs_lib_source = jobs_lib_path.read_text(encoding="utf-8")

        self.assertIn("runner startup state missing", jobs_lib_source)
        self.assertIn("for _runner_start in $(seq 1 100)", jobs_lib_source)
        self.assertIn("cat ${bashQuote(linuxPidWsl)}", jobs_lib_source)
        self.assertIn("bash -lc ${bashQuote(command)}", jobs_lib_source)
        self.assertIn("launch.sh", jobs_lib_source)
        self.assertIn("bash', toWslPath(launchScriptPath)", jobs_lib_source)
        self.assertNotIn("runner_pid=$!", jobs_lib_source)
        self.assertNotIn('wait "$runner_pid"; exit "$?"', jobs_lib_source)

    def test_stop_job_kills_wsl_child_process_group(self) -> None:
        jobs_lib_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        jobs_lib_source = jobs_lib_path.read_text(encoding="utf-8")

        self.assertIn("childPidPath", jobs_lib_source)
        self.assertIn("'child.pid'", jobs_lib_source)
        self.assertIn("setsid bash -lc ${bashQuote(command)}", jobs_lib_source)
        self.assertIn("stopWslPidFile", jobs_lib_source)
        self.assertIn("readRunnerPidFile", jobs_lib_source)
        self.assertIn("signalWslPidOrGroup", jobs_lib_source)
        self.assertIn("isWslPidOrGroupAlive", jobs_lib_source)
        self.assertIn("'/bin/kill'", jobs_lib_source)
        self.assertIn("`-${signal}`", jobs_lib_source)
        self.assertIn("killWslJobCommandMatches", jobs_lib_source)
        self.assertIn("shellSafeValue(job.expName)", jobs_lib_source)

    def test_inspect_model_cache_reports_snapshot_count_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            model_dir = cache_root / "models--Tongyi-MAI--Z-Image-Turbo"
            snapshot = model_dir / "snapshots" / "abc123"
            (model_dir / "refs").mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (model_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
            (snapshot / "config.json").write_text("12345", encoding="utf-8")
            (snapshot / "weights.bin").write_bytes(b"1234567890")

            summary = inspect_model_cache("Tongyi-MAI/Z-Image-Turbo", cache_root)

        self.assertTrue(summary["cached"])
        self.assertEqual(summary["snapshot_count"], 1)
        self.assertEqual(summary["size_bytes"], 21)
        self.assertTrue(summary["primary_snapshot"].endswith("abc123"))

    def test_build_zimage_command(self) -> None:
        command = build_accelerate_command(
            TrainingConfig(
                recipe_id="z-image-turbo-vlm",
                exp_name="test_run",
                max_train_steps=10,
                prefer_local_models=False,
            ),
            PROJECT_ROOT,
        )
        self.assertEqual(command["cwd"], str(PROJECT_ROOT / "z-image-turbo_self-distill-vlm"))
        self.assertIn("accelerate", command["args"])
        self.assertIn("--enable-gc", command["args"])
        self.assertIn("Tongyi-MAI/Z-Image-Turbo", command["args"])
        self.assertEqual(command["env"], {})

    def test_zimage_arguments_accept_local_smoke_memory_controls(self) -> None:
        module_path = PROJECT_ROOT / "z-image-turbo_self-distill-vlm" / "arguments.py"
        spec = importlib.util.spec_from_file_location("zimage_arguments_for_test", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        old_argv = sys.argv
        try:
            sys.argv = [
                "train_dopsd.py",
                "--exp-name",
                "zimage_smoke",
                "--low-vram",
                "--resolution-scale",
                "0.25",
                "--use-8bit-adam",
                "--no-save-samples",
                "--no-save-checkpoints",
            ]
            args = module.parse_args()
        finally:
            sys.argv = old_argv

        self.assertTrue(args.low_vram)
        self.assertEqual(args.resolution_scale, 0.25)
        self.assertTrue(args.use_8bit_adam)
        self.assertFalse(args.save_samples)
        self.assertFalse(args.save_checkpoints)

    def test_flux2_editing_arguments_accept_local_smoke_memory_controls(self) -> None:
        module_path = PROJECT_ROOT / "flux2-klein-edit-self-distill-gt-ref" / "arguments.py"
        spec = importlib.util.spec_from_file_location("flux2_editing_arguments_for_test", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        old_argv = sys.argv
        try:
            sys.argv = [
                "train_dopsd.py",
                "--exp-name",
                "flux2_editing_smoke",
                "--low-vram",
                "--resolution-scale",
                "0.25",
                "--use-8bit-adam",
                "--no-save-samples",
                "--no-save-checkpoints",
            ]
            args = module.parse_args()
        finally:
            sys.argv = old_argv

        self.assertTrue(args.low_vram)
        self.assertEqual(args.resolution_scale, 0.25)
        self.assertTrue(args.use_8bit_adam)
        self.assertFalse(args.save_samples)
        self.assertFalse(args.save_checkpoints)

    def test_recipe_command_can_enable_block_offload_guardrail(self) -> None:
        command = build_accelerate_command(
            TrainingConfig(
                recipe_id="flux2-klein-identity",
                exp_name="block_offload_smoke",
                launcher="python",
                use_deepspeed=False,
                low_vram=True,
                block_offload=True,
                block_offload_num_blocks=2,
                prefer_local_models=False,
            ),
            PROJECT_ROOT,
        )

        self.assertIn("--low-vram", command["args"])
        self.assertIn("--block-offload", command["args"])
        block_index = command["args"].index("--block-offload-num-blocks")
        self.assertEqual(command["args"][block_index + 1], "2")
        self.assertIn("--block-offload --block-offload-num-blocks 2", command["display"])

    def test_dopsd_training_scripts_expose_diffusers_group_block_offload(self) -> None:
        for relative in (
            "flux2-klein_self-distill-edit",
            "flux2-klein-edit-self-distill-gt-ref",
            "z-image-turbo_self-distill-vlm",
        ):
            with self.subTest(relative=relative):
                args_source = (PROJECT_ROOT / relative / "arguments.py").read_text(encoding="utf-8")
                train_source = (PROJECT_ROOT / relative / "train_dopsd.py").read_text(encoding="utf-8")

                self.assertIn("--block-offload", args_source)
                self.assertIn("--block-offload-num-blocks", args_source)
                self.assertIn("apply_group_offloading", train_source)
                self.assertIn("enable_transformer_block_offload", train_source)
                self.assertIn('offload_type="block_level"', train_source)
                self.assertIn("Transformer block offload requested for final sampling only", train_source)
                self.assertIn("if args.block_offload and global_step == args.max_train_steps", train_source)

    def test_flux2_block_offload_is_enabled_after_pipeline_device_prepare(self) -> None:
        for relative in (
            "flux2-klein_self-distill-edit",
            "flux2-klein-edit-self-distill-gt-ref",
        ):
            with self.subTest(relative=relative):
                source = (PROJECT_ROOT / relative / "train_dopsd.py").read_text(encoding="utf-8")
                sample_index = source.find("# visualize samples")
                prepare_index = source.find(
                    "prepare_pipeline_for_sampling(pipeline, accelerator.device, inference_dtype, vae_dtype)",
                    sample_index,
                )
                offload_index = source.find(
                    "enable_transformer_block_offload(",
                    sample_index,
                )

                self.assertGreaterEqual(sample_index, 0)
                self.assertGreater(
                    prepare_index,
                    sample_index,
                    "FLUX2 final sampling should prepare pipeline devices before block offload hooks are attached",
                )
                self.assertGreater(
                    offload_index,
                    prepare_index,
                    "FLUX2 final sampling must not call pipeline.to() after the transformer is group offloaded",
                )

    def test_flux2_block_offload_sampling_decodes_latents_after_pipeline_hooks(self) -> None:
        for relative in (
            "flux2-klein_self-distill-edit",
            "flux2-klein-edit-self-distill-gt-ref",
        ):
            with self.subTest(relative=relative):
                source = (PROJECT_ROOT / relative / "train_dopsd.py").read_text(encoding="utf-8")

                self.assertIn("def sample_flux2_with_optional_latent_decode", source)
                self.assertIn('output_type="latent"', source)
                self.assertIn("pipeline.image_processor.postprocess", source)
                self.assertIn("decode_latents_after_transformer_hooks=args.block_offload", source)

    def test_flux2_prompt_encoding_limits_qwen_logits_memory(self) -> None:
        for relative in (
            "flux2-klein_self-distill-edit",
            "flux2-klein-edit-self-distill-gt-ref",
        ):
            with self.subTest(relative=relative):
                source = (PROJECT_ROOT / relative / "train_dopsd.py").read_text(encoding="utf-8")
                encode_start = source.find("def _encode_prompt(")
                encode_end = source.find("def _patchify_latents", encode_start)
                encode_source = source[encode_start:encode_end]

                self.assertIn(
                    "logits_to_keep=1",
                    encode_source,
                    "FLUX2 prompt encoding only needs hidden states; limiting Qwen logits avoids full lm_head memory spikes",
                )
                self.assertIn("del output, input_ids, attention_mask", encode_source)

    def test_flux2_identity_block_offload_wraps_teacher_condition_sample(self) -> None:
        source = (PROJECT_ROOT / "flux2-klein_self-distill-edit" / "train_dopsd.py").read_text(
            encoding="utf-8"
        )
        sample_index = source.find("# sample multistep images for comparison")
        teacher_index = source.find("teacher_test_images = []", sample_index)
        offload_index = source.find("enable_transformer_block_offload(", sample_index)

        self.assertGreaterEqual(sample_index, 0)
        self.assertGreater(teacher_index, sample_index)
        self.assertGreater(offload_index, sample_index)
        self.assertLess(
            offload_index,
            teacher_index,
            "FLUX2 Identity should attach block offload before teacher image-condition VAE encode",
        )
        teacher_block = source[teacher_index : source.find("images_t = torch.cat", teacher_index)]
        self.assertIn(
            "decode_latents_after_transformer_hooks=args.block_offload",
            teacher_block,
            "Teacher condition sampling should use the after-hook latent decode path when block offload is enabled",
        )

    def test_flux2_identity_post_train_sampling_releases_training_tensors_before_teacher_sample(self) -> None:
        script_path = PROJECT_ROOT / "flux2-klein_self-distill-edit" / "train_dopsd.py"
        source = script_path.read_text(encoding="utf-8")

        visualize_index = source.find("# visualize samples")
        trajectory_index = source.find("save_student_teacher_trajectory(", visualize_index)
        release_index = source.find("del student_x0_traj", trajectory_index)
        teacher_sample_index = source.find("teacher_test_images = []", trajectory_index)

        self.assertGreaterEqual(visualize_index, 0)
        self.assertGreaterEqual(trajectory_index, 0)
        self.assertGreater(
            release_index,
            trajectory_index,
            "FLUX2 Identity should release trajectory/training tensors after trajectory PNGs are saved",
        )
        self.assertLess(
            release_index,
            teacher_sample_index,
            "training tensor release must happen before the teacher image-condition VAE encode sample",
        )
        release_block = source[release_index:teacher_sample_index]
        self.assertIn("teacher_image_latents", release_block)
        self.assertIn("prompt_embeds", release_block)
        self.assertIn("latents_begin", release_block)
        self.assertIn("free_cuda_memory()", release_block)

    def test_flux2_identity_arguments_accept_final_sample_subprocess_mode(self) -> None:
        module_path = PROJECT_ROOT / "flux2-klein_self-distill-edit" / "arguments.py"
        spec = importlib.util.spec_from_file_location("flux2_identity_arguments_for_test", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        old_argv = sys.argv
        try:
            sys.argv = [
                "train_dopsd.py",
                "--exp-name",
                "flux2_identity_sampler_subprocess",
                "--final-sample-mode",
                "subprocess",
                "--final-sample-resolution-scale",
                "0.5",
                "--no-final-sampler-cpu-offload",
                "--no-tiled-vae",
                "--vae-tile-size",
                "96",
                "--vae-tile-overlap",
                "0.125",
            ]
            args = module.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.final_sample_mode, "subprocess")
        self.assertEqual(args.final_sample_resolution_scale, 0.5)
        self.assertFalse(args.final_sampler_cpu_offload)
        self.assertFalse(args.tiled_vae)
        self.assertEqual(args.vae_tile_size, 96)
        self.assertEqual(args.vae_tile_overlap, 0.125)

    def test_flux2_identity_inline_sample_resolution_can_be_scaled_for_low_vram(self) -> None:
        args_path = PROJECT_ROOT / "flux2-klein_self-distill-edit" / "arguments.py"
        train_path = PROJECT_ROOT / "flux2-klein_self-distill-edit" / "train_dopsd.py"
        smoke_path = PROJECT_ROOT / "scripts" / "run_flux2_smoke.sh"
        args_source = args_path.read_text(encoding="utf-8")

        self.assertIn("--sample-resolution-scale", args_source)

        spec = importlib.util.spec_from_file_location("flux2_identity_arguments_sample_scale_test", args_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        old_argv = sys.argv
        try:
            sys.argv = [
                "train_dopsd.py",
                "--exp-name",
                "flux2_identity_sample_scale",
                "--sample-resolution-scale",
                "0.5",
            ]
            args = module.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.sample_resolution_scale, 0.5)

        train_source = train_path.read_text(encoding="utf-8")
        smoke_source = smoke_path.read_text(encoding="utf-8")

        self.assertIn("def scale_sample_size", train_source)
        self.assertIn("args.sample_resolution_scale", train_source)
        self.assertIn("sample_h, sample_w = scale_sample_size", train_source)
        self.assertIn("height=sample_h", train_source)
        self.assertIn("width=sample_w", train_source)
        self.assertIn("Saved original sample images", train_source)
        self.assertIn('SAMPLE_RESOLUTION_SCALE="${SAMPLE_RESOLUTION_SCALE:-1.0}"', smoke_source)
        self.assertIn('--sample-resolution-scale "$SAMPLE_RESOLUTION_SCALE"', smoke_source)

    def test_flux2_identity_block_offload_final_samples_defer_to_subprocess(self) -> None:
        train_source = (PROJECT_ROOT / "flux2-klein_self-distill-edit" / "train_dopsd.py").read_text(
            encoding="utf-8"
        )
        sampler_path = PROJECT_ROOT / "flux2-klein_self-distill-edit" / "sample_flux2_final.py"
        smoke_source = (PROJECT_ROOT / "scripts" / "run_flux2_smoke.sh").read_text(encoding="utf-8")

        self.assertTrue(sampler_path.exists(), "FLUX2 Identity needs a separate final sampler script")
        self.assertIn("def should_defer_final_sampling", train_source)
        self.assertIn("def write_final_sampler_request", train_source)
        self.assertIn("def scale_final_sample_size", train_source)
        self.assertIn("final_sampler_payload", train_source)
        self.assertIn("sample_flux2_final.py", train_source)
        self.assertIn("args.final_sample_mode", train_source)
        self.assertIn("args.final_sample_resolution_scale", train_source)
        self.assertIn("unwrap_model(gen_model, accelerator).save_pretrained", train_source)
        self.assertIn('"requested_block_offload": bool(args.block_offload)', train_source)
        self.assertIn('"block_offload": False', train_source)
        self.assertIn("del pipeline, gen_model", train_source)
        self.assertIn("Final sample generation request ready for runner", train_source)
        self.assertNotIn("os.execv(", train_source)
        self.assertNotIn("launch_final_sampler_subprocess(final_sampler_payload", train_source)
        self.assertIn('FINAL_SAMPLE_MODE="${FINAL_SAMPLE_MODE:-auto}"', smoke_source)
        self.assertIn('FINAL_SAMPLE_RESOLUTION_SCALE="${FINAL_SAMPLE_RESOLUTION_SCALE:-0.5}"', smoke_source)
        self.assertIn('--final-sample-mode "$FINAL_SAMPLE_MODE"', smoke_source)
        self.assertIn('--final-sample-resolution-scale "$FINAL_SAMPLE_RESOLUTION_SCALE"', smoke_source)
        self.assertIn("run_deferred_final_sampler", smoke_source)
        self.assertIn('python sample_flux2_final.py --request "$final_sampler_request"', smoke_source)

    def test_flux2_identity_final_sampler_loads_adapters_and_writes_standard_samples(self) -> None:
        sampler_path = PROJECT_ROOT / "flux2-klein_self-distill-edit" / "sample_flux2_final.py"
        self.assertTrue(sampler_path.exists())
        source = sampler_path.read_text(encoding="utf-8")

        self.assertIn("PeftModel.from_pretrained", source)
        self.assertIn('adapter_name="student"', source)
        self.assertIn('load_adapter(str(teacher_dir), adapter_name="teacher"', source)
        self.assertIn('samples_step_{global_step}_teacher.png', source)
        self.assertIn('samples_step_{global_step}_student.png', source)
        self.assertIn("sample_flux2_with_optional_latent_decode", source)
        self.assertIn("enable_transformer_block_offload", source)
        self.assertIn("display_height", source)
        self.assertIn('request.get("source_height", height)', source)
        self.assertIn("append_log(request", source)
        self.assertIn("Training completed.", source)

    def test_flux2_identity_final_sampler_uses_tiled_vae_for_image_conditioning(self) -> None:
        args_source = (PROJECT_ROOT / "flux2-klein_self-distill-edit" / "arguments.py").read_text(
            encoding="utf-8"
        )
        train_source = (PROJECT_ROOT / "flux2-klein_self-distill-edit" / "train_dopsd.py").read_text(
            encoding="utf-8"
        )
        sampler_source = (
            PROJECT_ROOT / "flux2-klein_self-distill-edit" / "sample_flux2_final.py"
        ).read_text(encoding="utf-8")

        self.assertIn("--tiled-vae", args_source)
        self.assertIn("--vae-tile-size", args_source)
        self.assertIn("--vae-tile-overlap", args_source)
        self.assertIn('"tiled_vae": bool(args.tiled_vae)', train_source)
        self.assertIn('"vae_tile_size": int(args.vae_tile_size)', train_source)
        self.assertIn('"vae_tile_overlap": float(args.vae_tile_overlap)', train_source)
        self.assertIn("def configure_tiled_vae", sampler_source)
        self.assertIn('request.get("tiled_vae", True)', sampler_source)
        self.assertIn("vae.enable_slicing()", sampler_source)
        self.assertIn("vae.enable_tiling()", sampler_source)
        self.assertIn("tile_sample_min_size", sampler_source)
        self.assertIn("tile_latent_min_size", sampler_source)
        self.assertIn("tile_overlap_factor", sampler_source)
        self.assertIn("Tiled VAE enabled", sampler_source)

        configure_index = sampler_source.find("configure_tiled_vae(pipeline.vae")
        cpu_offload_index = sampler_source.find("configure_pipeline_cpu_offload(", configure_index)
        self.assertGreaterEqual(configure_index, 0)
        self.assertGreater(cpu_offload_index, configure_index)

    def test_flux2_identity_inline_sampling_configures_tiled_vae_before_first_decode(self) -> None:
        train_source = (PROJECT_ROOT / "flux2-klein_self-distill-edit" / "train_dopsd.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("def configure_tiled_vae", train_source)
        self.assertIn("vae.enable_tiling()", train_source)
        self.assertIn("tile_sample_min_size", train_source)
        self.assertIn("tile_latent_min_size", train_source)
        self.assertIn("tile_overlap_factor", train_source)
        self.assertIn("configure_tiled_vae(pipeline.vae, args, logger)", train_source)

        configure_index = train_source.find("configure_tiled_vae(pipeline.vae, args, logger)")
        sample_branch_index = train_source.find("if args.save_samples:", train_source.find("progress_bar"))
        first_pipeline_decode_index = train_source.find("images = pipeline(", sample_branch_index)
        self.assertGreaterEqual(configure_index, 0)
        self.assertGreater(sample_branch_index, configure_index)
        self.assertGreater(first_pipeline_decode_index, configure_index)

    def test_flux2_identity_final_sampler_uses_model_cpu_offload(self) -> None:
        args_source = (PROJECT_ROOT / "flux2-klein_self-distill-edit" / "arguments.py").read_text(
            encoding="utf-8"
        )
        train_source = (PROJECT_ROOT / "flux2-klein_self-distill-edit" / "train_dopsd.py").read_text(
            encoding="utf-8"
        )
        sampler_source = (
            PROJECT_ROOT / "flux2-klein_self-distill-edit" / "sample_flux2_final.py"
        ).read_text(encoding="utf-8")

        self.assertIn("--final-sampler-cpu-offload", args_source)
        self.assertIn('"final_sampler_cpu_offload": bool(args.final_sampler_cpu_offload)', train_source)
        self.assertIn("def configure_pipeline_cpu_offload", sampler_source)
        self.assertIn('request.get("final_sampler_cpu_offload", True)', sampler_source)
        self.assertIn("pipeline.enable_model_cpu_offload(device=device)", sampler_source)
        self.assertIn("Final sampler model CPU offload enabled", sampler_source)
        self.assertIn("pipeline.vae.to(dtype=vae_dtype)", sampler_source)
        self.assertIn("pipeline.vae.to(device, dtype=vae_dtype)", sampler_source)

    def test_smoke_scripts_allow_env_controlled_artifact_writes(self) -> None:
        for script_name in (
            "run_flux2_smoke.sh",
            "run_flux2_editing_smoke.sh",
            "run_zimage_smoke.sh",
        ):
            with self.subTest(script_name=script_name):
                source = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")

                self.assertIn('SAVE_SAMPLES="${SAVE_SAMPLES:-0}"', source)
                self.assertIn('SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"', source)
                self.assertIn('SAMPLE_STEPS="${SAMPLE_STEPS:-999}"', source)
                self.assertIn('CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-999}"', source)
                self.assertIn('--sample-steps "$SAMPLE_STEPS"', source)
                self.assertIn('--checkpoint-steps "$CHECKPOINT_STEPS"', source)
                self.assertIn("--save-samples", source)
                self.assertIn("--no-save-samples", source)
                self.assertIn("--save-checkpoints", source)
                self.assertIn("--no-save-checkpoints", source)

    def test_smoke_scripts_allow_env_controlled_train_steps(self) -> None:
        for script_name in (
            "run_flux2_smoke.sh",
            "run_flux2_editing_smoke.sh",
            "run_zimage_smoke.sh",
        ):
            with self.subTest(script_name=script_name):
                source = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")

                self.assertIn('MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1}"', source)
                self.assertIn('--max-train-steps "$MAX_TRAIN_STEPS"', source)

    def test_new_job_ui_surfaces_block_offload_controls(self) -> None:
        page_path = PROJECT_ROOT / "trainer-ui" / "src" / "app" / "jobs" / "new" / "page.tsx"
        wizard_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "NewJobWizard.tsx"
        i18n_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx"
        source = page_path.read_text(encoding="utf-8") + wizard_path.read_text(encoding="utf-8")
        i18n_source = i18n_path.read_text(encoding="utf-8")

        self.assertIn("blockOffload", source)
        self.assertIn("transformerBlocks", source)
        self.assertIn("Block offload", i18n_source)
        self.assertIn("块卸载", i18n_source)
        self.assertIn("Transformer blocks", i18n_source)
        self.assertIn("--block-offload", source)
        self.assertIn("--block-offload-num-blocks", source)

    def test_flux2_sampling_uses_prepared_transformer(self) -> None:
        script_path = PROJECT_ROOT / "flux2-klein_self-distill-edit" / "train_dopsd.py"
        source = script_path.read_text(encoding="utf-8")

        prepare_index = source.find("accelerator.prepare(gen_model")
        rebind_index = source.find("pipeline.transformer = gen_model", prepare_index)
        sample_device_index = source.find("prepare_pipeline_for_sampling(", rebind_index)
        first_sample_call_index = source.find("images = pipeline(", prepare_index)

        self.assertGreaterEqual(prepare_index, 0)
        self.assertGreater(
            rebind_index,
            prepare_index,
            "pipeline.transformer must be rebound to the prepared gen_model before sampling",
        )
        self.assertLess(
            rebind_index,
            first_sample_call_index,
            "pipeline sampling should use the device-prepared transformer",
        )
        self.assertGreater(
            sample_device_index,
            rebind_index,
            "pipeline execution device must be set before sampling",
        )
        self.assertLess(
            sample_device_index,
            first_sample_call_index,
            "pipeline execution device should be set before the first sample block",
        )
        self.assertIn("pipeline.to(device)", source)

    def test_flux2_editing_low_vram_smoke_path_matches_identity_guardrails(self) -> None:
        script_path = PROJECT_ROOT / "flux2-klein-edit-self-distill-gt-ref" / "train_dopsd.py"
        source = script_path.read_text(encoding="utf-8")

        prepare_index = source.find("accelerator.prepare(gen_model")
        rebind_index = source.find("pipeline.transformer = gen_model", prepare_index)
        low_vram_index = source.find("if args.low_vram:")
        checkpoint_guard_index = source.find("if args.save_checkpoints")
        sample_guard_index = source.find("if args.save_samples")

        self.assertIn("def scale_resolutions", source)
        self.assertIn("def offload_frozen_conditioners", source)
        self.assertIn("def prepare_pipeline_for_sampling", source)
        self.assertIn("zero2_plugin_a = None", source)
        self.assertIn("if zero2_plugin_a is not None:", source)
        self.assertIn("gen_model, optimizer_gen = accelerator.prepare(gen_model, optimizer_gen)", source)
        self.assertIn(
            "target_resolutions = scale_resolutions(parse_ratios(select_ratio), args.resolution_scale, args.target_resolution)",
            source,
        )
        self.assertGreaterEqual(prepare_index, 0)
        self.assertGreater(
            rebind_index,
            prepare_index,
            "FLUX2 Editing pipeline sampling must use the prepared transformer",
        )
        self.assertGreaterEqual(low_vram_index, 0)
        self.assertGreaterEqual(checkpoint_guard_index, 0)
        self.assertGreaterEqual(sample_guard_index, 0)
        self.assertIn("offload_frozen_conditioners(pipeline)", source)
        self.assertIn("Sample generation disabled", source)

    def test_flux2_editing_sampling_resizes_condition_images_before_vae_encode(self) -> None:
        script_path = PROJECT_ROOT / "flux2-klein-edit-self-distill-gt-ref" / "train_dopsd.py"
        source = script_path.read_text(encoding="utf-8")

        helper_start = source.find("def load_rgb_image")
        sample_start = source.find("def run_pipeline_per_edit_sample")
        sample_end = source.find("@torch.no_grad()", sample_start)
        self.assertGreaterEqual(helper_start, 0)
        self.assertGreaterEqual(sample_start, 0)
        self.assertGreater(sample_end, sample_start)

        helper_source = source[helper_start:sample_start]
        sample_source = source[sample_start:sample_end]
        self.assertIn("target_size=None", helper_source)
        self.assertIn("process_image_for_sampling", helper_source)
        self.assertIn("load_rgb_image(image_path, target_size=(width, height))", sample_source)

    def test_flux2_editing_inline_sample_resolution_can_be_scaled_for_low_vram(self) -> None:
        args_path = PROJECT_ROOT / "flux2-klein-edit-self-distill-gt-ref" / "arguments.py"
        train_path = PROJECT_ROOT / "flux2-klein-edit-self-distill-gt-ref" / "train_dopsd.py"
        smoke_path = PROJECT_ROOT / "scripts" / "run_flux2_editing_smoke.sh"
        args_source = args_path.read_text(encoding="utf-8")

        self.assertIn("--sample-resolution-scale", args_source)

        spec = importlib.util.spec_from_file_location("flux2_editing_arguments_sample_scale_test", args_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        old_argv = sys.argv
        try:
            sys.argv = [
                "train_dopsd.py",
                "--exp-name",
                "flux2_editing_sample_scale",
                "--sample-resolution-scale",
                "0.5",
            ]
            args = module.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.sample_resolution_scale, 0.5)

        train_source = train_path.read_text(encoding="utf-8")
        smoke_source = smoke_path.read_text(encoding="utf-8")

        self.assertIn("def scale_sample_size", train_source)
        self.assertIn("args.sample_resolution_scale", train_source)
        self.assertIn("sample_h, sample_w = scale_sample_size", train_source)
        self.assertIn("height=sample_h", train_source)
        self.assertIn("width=sample_w", train_source)
        self.assertIn("Saved original sample images", train_source)
        self.assertIn('SAMPLE_RESOLUTION_SCALE="${SAMPLE_RESOLUTION_SCALE:-1.0}"', smoke_source)
        self.assertIn('--sample-resolution-scale "$SAMPLE_RESOLUTION_SCALE"', smoke_source)

    def test_flux2_editing_post_train_sampling_releases_training_tensors(self) -> None:
        script_path = PROJECT_ROOT / "flux2-klein-edit-self-distill-gt-ref" / "train_dopsd.py"
        source = script_path.read_text(encoding="utf-8")

        visualize_index = source.find("# visualize samples")
        trajectory_index = source.find("save_student_teacher_trajectory(", visualize_index)
        release_index = source.find("del student_x0_traj", trajectory_index)
        sample_index = source.find('gen_model.set_adapter("student")', trajectory_index)
        self.assertGreaterEqual(visualize_index, 0)
        self.assertGreaterEqual(trajectory_index, 0)
        self.assertGreater(
            release_index,
            trajectory_index,
            "FLUX2 Editing should release trajectory/training tensors after trajectory PNGs are saved",
        )
        self.assertLess(
            release_index,
            sample_index,
            "training tensor release must happen before the post-train pipeline sample",
        )
        release_block = source[release_index:sample_index]
        self.assertIn("student_image_latents", release_block)
        self.assertIn("teacher_image_latents", release_block)
        self.assertIn("prompt_embeds", release_block)
        self.assertIn("free_cuda_memory()", release_block)

    def test_flux2_editing_inline_sampling_configures_tiled_vae_before_first_decode(self) -> None:
        args_source = (PROJECT_ROOT / "flux2-klein-edit-self-distill-gt-ref" / "arguments.py").read_text(
            encoding="utf-8"
        )
        train_source = (PROJECT_ROOT / "flux2-klein-edit-self-distill-gt-ref" / "train_dopsd.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("--tiled-vae", args_source)
        self.assertIn("--vae-tile-size", args_source)
        self.assertIn("--vae-tile-overlap", args_source)
        self.assertIn("def configure_tiled_vae", train_source)
        self.assertIn("vae.enable_tiling()", train_source)
        self.assertIn("tile_sample_min_size", train_source)
        self.assertIn("tile_latent_min_size", train_source)
        self.assertIn("tile_overlap_factor", train_source)
        self.assertIn("configure_tiled_vae(pipeline.vae, args, logger)", train_source)

        configure_index = train_source.find("configure_tiled_vae(pipeline.vae, args, logger)")
        sample_branch_index = train_source.find("if args.save_samples:", train_source.find("progress_bar"))
        first_sample_decode_index = train_source.find("run_pipeline_per_edit_sample(", sample_branch_index)
        self.assertGreaterEqual(configure_index, 0)
        self.assertGreater(sample_branch_index, configure_index)
        self.assertGreater(first_sample_decode_index, configure_index)

    def test_flux2_editing_smoke_script_uses_low_vram_defaults(self) -> None:
        script_path = PROJECT_ROOT / "scripts" / "run_flux2_editing_smoke.sh"
        self.assertTrue(script_path.exists())
        source = script_path.read_text(encoding="utf-8")

        self.assertIn("flux2-klein-edit-self-distill-gt-ref", source)
        self.assertIn("dataset/interaction/data.jsonl", source)
        self.assertIn("--resolution-scale", source)
        self.assertIn("--low-vram", source)
        self.assertIn("--no-save-samples", source)
        self.assertIn("--no-save-checkpoints", source)
        self.assertIn("--use-8bit-adam", source)

    def test_jobs_runner_wires_flux2_editing_smoke_script(self) -> None:
        jobs_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        source = jobs_path.read_text(encoding="utf-8")

        self.assertIn("run_flux2_editing_smoke.sh", source)
        self.assertIn("commandForFlux2EditingSmoke", source)
        self.assertIn("job.recipeId === 'flux2-klein-editing'", source)
        self.assertIn("productionProfileForRecipe(base.id)", source)
        self.assertIn("profile.runnerScript", source)
        self.assertNotIn(
            "Runner start is currently verified only for FLUX2 Identity and Z-Image low-VRAM jobs.",
            source,
        )

    def test_jobs_ui_can_open_output_folder_safely(self) -> None:
        jobs_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        route_path = PROJECT_ROOT / "trainer-ui" / "src" / "app" / "api" / "jobs" / "[id]" / "open" / "route.ts"
        table_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "JobsTable.tsx"

        jobs_source = jobs_path.read_text(encoding="utf-8")
        table_source = table_path.read_text(encoding="utf-8")

        self.assertIn("export async function openJobOutputFolder", jobs_source)
        self.assertIn("ensureProjectChildPath", jobs_source)
        self.assertIn("Output folder does not exist yet", jobs_source)
        self.assertIn("spawn('explorer.exe'", jobs_source)
        self.assertIn("detached: true", jobs_source)

        self.assertTrue(route_path.exists())
        route_source = route_path.read_text(encoding="utf-8")
        self.assertIn("openJobOutputFolder", route_source)
        self.assertIn("POST", route_source)

        self.assertIn("FolderOpen", table_source)
        self.assertIn("openOutput", table_source)
        self.assertIn("`/api/jobs/${job.id}/open`", table_source)

    def test_failed_run_summary_extracts_runner_error_reason(self) -> None:
        run_dir = (
            PROJECT_ROOT
            / "trainer-data"
            / "smoke-runs"
            / "flux2_editing_sample_smoke_1step_lowvram_20260622052603"
        )
        runner_err = run_dir.parent / f"{run_dir.name}.runner.err.log"
        self.assertTrue(runner_err.exists(), "real failed-run stderr fixture should exist")

        summary = summarize_run(run_dir).to_dict()

        self.assertEqual(summary["status"], "failed")
        self.assertIn("out of memory", summary["failure_reason"].lower())
        self.assertTrue(summary["error_tail"])
        self.assertTrue(any("RuntimeError" in line or "out of memory" in line for line in summary["error_tail"]))

    def test_jobs_ui_surfaces_failure_reason_from_runner_error_logs(self) -> None:
        jobs_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        table_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "JobsTable.tsx"

        jobs_source = jobs_path.read_text(encoding="utf-8")
        table_source = table_path.read_text(encoding="utf-8")

        self.assertIn("failureReason", jobs_source)
        self.assertIn("errorTail", jobs_source)
        self.assertIn(".runner.err.log", jobs_source)
        self.assertIn("seedKnownFailureJobs", jobs_source)
        self.assertIn("Seeded failure evidence jobs must be cloned before rerun", jobs_source)

        self.assertIn("AlertTriangle", table_source)
        self.assertIn("FailureDiagnostics", table_source)
        self.assertIn("failureReason", table_source)
        self.assertIn("errorTail", table_source)
        self.assertIn("job.source !== 'seeded-failure'", table_source)

    def test_jobs_runner_recovery_checks_linux_pid_before_marking_failed(self) -> None:
        jobs_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        source = jobs_path.read_text(encoding="utf-8")

        self.assertIn("isLinuxPidAlive", source)
        self.assertIn("runnerAliveInWsl", source)
        self.assertIn("Runner process monitor detached", source)
        self.assertIn("linuxPidFileExists", source)
        self.assertIn("Linux PID check is inconclusive", source)
        self.assertIn("linuxPidPath", source)
        self.assertIn("timeout: 15000", source)

        pid_check_index = source.find("const runnerAliveInWsl")
        fail_note_index = source.find("Runner process exited before writing completion state")
        self.assertGreaterEqual(pid_check_index, 0)
        self.assertGreater(
            fail_note_index,
            pid_check_index,
            "refreshRunnerState should check WSL Linux PID before marking a job failed",
        )

    def test_jobs_runner_launch_waits_for_linux_startup_and_captures_wsl_stderr(self) -> None:
        jobs_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        source = jobs_path.read_text(encoding="utf-8")

        self.assertIn("RUNNER_STARTUP_TIMEOUT_MS", source)
        self.assertIn("launchDetachedWslRunner", source)
        self.assertIn("waitForRunnerStartup", source)
        self.assertIn("runner.stdout.log", source)
        self.assertIn("runner.stderr.log", source)
        self.assertIn("windows.pid", source)
        self.assertIn("wsl.exe", source)
        self.assertIn("spawn('wsl.exe'", source)
        self.assertIn("buildDetachedWslLaunchCommand", source)
        self.assertIn("launchScriptPath", source)
        self.assertIn("bash', toWslPath(launchScriptPath)", source)
        self.assertIn("nohup bash", source)
        self.assertIn("runner startup state missing", source)
        self.assertIn("for _runner_start in $(seq 1 100)", source)
        self.assertIn("cat ${bashQuote(linuxPidWsl)}", source)
        self.assertIn("kill -0", source)
        self.assertIn("].join('\\n')", source)
        self.assertIn("Runner failed to write Linux startup state", source)
        self.assertIn("Runner startup stderr", source)
        self.assertIn("readCombinedLogTail(runnerLogPaths)", source)
        self.assertIn("stdio: ['ignore', 'pipe', 'pipe']", source)

    def test_jobs_ledger_uses_atomic_write_and_retries_parse_failures(self) -> None:
        jobs_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        source = jobs_path.read_text(encoding="utf-8")

        self.assertIn("readJobsLedgerWithRetry", source)
        self.assertIn("Invalid jobs ledger JSON", source)
        self.assertIn("uFEFF", source)
        self.assertNotIn("catch {\n    return { version: 1, jobs: [] };", source)
        self.assertIn(".tmp-", source)
        self.assertIn("fs.rename", source)

    def test_jobs_ui_can_open_full_log_panel(self) -> None:
        jobs_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        route_path = PROJECT_ROOT / "trainer-ui" / "src" / "app" / "api" / "jobs" / "[id]" / "logs" / "route.ts"
        table_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "JobsTable.tsx"

        jobs_source = jobs_path.read_text(encoding="utf-8")
        table_source = table_path.read_text(encoding="utf-8")

        self.assertTrue(route_path.exists())
        route_source = route_path.read_text(encoding="utf-8")
        self.assertIn("getJobLogs", route_source)
        self.assertIn("GET", route_source)

        self.assertIn("export async function getJobLogs", jobs_source)
        self.assertIn("MAX_LOG_LINES", jobs_source)
        self.assertIn(".runner.err.log", jobs_source)
        self.assertIn("combined", jobs_source)

        self.assertIn("ScrollText", table_source)
        self.assertIn("LogPanel", table_source)
        self.assertIn("`/api/jobs/${job.id}/logs`", table_source)
        self.assertIn("t('fullLogs')", table_source)

    def test_jobs_log_stream_route_uses_sse_and_jobs_ui_subscribes_to_eventsource(self) -> None:
        route_path = (
            PROJECT_ROOT
            / "trainer-ui"
            / "src"
            / "app"
            / "api"
            / "jobs"
            / "[id]"
            / "logs"
            / "stream"
            / "route.ts"
        )
        table_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "JobsTable.tsx"

        self.assertTrue(route_path.exists(), "running job logs should have a push-style SSE route")
        route_source = route_path.read_text(encoding="utf-8")
        table_source = table_path.read_text(encoding="utf-8")

        self.assertIn("text/event-stream", route_source)
        self.assertIn("ReadableStream", route_source)
        self.assertIn("event: snapshot", route_source)
        self.assertIn("event: append", route_source)
        self.assertIn("event: heartbeat", route_source)
        self.assertIn("AbortSignal", route_source)
        self.assertIn("getJobLogs", route_source)
        self.assertIn("setInterval", route_source)
        self.assertIn("clearInterval", route_source)

        self.assertIn("new EventSource(`/api/jobs/${job.id}/logs/stream`)", table_source)
        self.assertIn("addEventListener('snapshot'", table_source)
        self.assertIn("addEventListener('append'", table_source)
        self.assertIn("eventSource.close()", table_source)
        self.assertIn("streaming", table_source)
        self.assertIn("t('liveStream')", table_source)
        self.assertIn("slice(-300)", table_source)

    def test_zimage_low_vram_sampling_uses_cpu_prompt_embeds(self) -> None:
        script_path = PROJECT_ROOT / "z-image-turbo_self-distill-vlm" / "train_dopsd.py"
        source = script_path.read_text(encoding="utf-8")

        prepare_index = source.find("accelerator.prepare(gen_model")
        rebind_index = source.find("pipeline.transformer = gen_model", prepare_index)
        initial_sample_start = source.find("if args.save_samples:", prepare_index)
        initial_sample_end = source.find("grad_norm = 0", initial_sample_start)
        final_sample_start = source.find("# visualize samples", initial_sample_end)
        final_sample_end = source.find("progress_bar.set_postfix", final_sample_start)

        self.assertGreaterEqual(prepare_index, 0)
        self.assertGreater(
            rebind_index,
            prepare_index,
            "pipeline.transformer must be rebound to the prepared gen_model before Z-Image sampling",
        )
        self.assertLess(
            rebind_index,
            initial_sample_start,
            "Z-Image sampling should use the prepared transformer",
        )
        self.assertIn(
            "def sample_zimage_from_prompt_embeds",
            source,
            "Z-Image low-VRAM sampling needs an explicit-device helper instead of pipeline device inference",
        )
        self.assertIn(
            "pipeline._interrupt = False",
            source,
            "Explicit Z-Image sampling should initialize the same interrupt state as the diffusers pipeline call",
        )
        self.assertGreaterEqual(initial_sample_start, 0)
        self.assertGreaterEqual(initial_sample_end, 0)
        self.assertGreaterEqual(final_sample_start, 0)
        self.assertGreaterEqual(final_sample_end, 0)

        for block_name, block in (
            ("initial sample block", source[initial_sample_start:initial_sample_end]),
            ("student sample block", source[final_sample_start:final_sample_end]),
        ):
            self.assertNotIn(
                "pipeline.text_encoder.to(accelerator.device",
                block,
                f"{block_name} must not move the Z-Image text encoder back to CUDA in low-VRAM sampling",
            )
            self.assertIn(
                "prompt_embeds_test",
                block,
                f"{block_name} should use precomputed text prompt embeddings",
            )
            self.assertNotIn(
                "prompt=test_prompts",
                block,
                f"{block_name} should avoid pipeline-side prompt encoding in low-VRAM sampling",
            )
            self.assertNotIn(
                " = pipeline(",
                block,
                f"{block_name} should avoid Z-Image pipeline __call__ device inference in low-VRAM sampling",
            )
            self.assertIn(
                "sample_zimage_from_prompt_embeds(",
                block,
                f"{block_name} should use explicit-device Z-Image sampling",
            )

    def test_parse_flux2_smoke_outputs(self) -> None:
        run_dir = PROJECT_ROOT / "trainer-data" / "smoke-runs" / "flux2_identity_smoke_1step_lowvram_python"
        summary = summarize_run(run_dir)
        self.assertEqual(summary.status, "completed")
        self.assertTrue(summary.has_args)
        self.assertTrue(summary.has_log)
        self.assertTrue(summary.has_loss_log)
        self.assertEqual(summary.latest_step, 1)
        self.assertAlmostEqual(summary.latest_loss or 0, 1.9178619384765625)
        self.assertEqual(summary.trainable_params, 983040)

        points = parse_loss_jsonl(run_dir / "loss_log" / "loss_gen_log.jsonl")
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].step, 1)

    def test_summarize_run_lists_samples_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            samples = run_dir / "samples"
            checkpoints = run_dir / "checkpoints"
            samples.mkdir()
            checkpoints.mkdir()
            (samples / "step-0001.png").write_bytes(b"image")
            (checkpoints / "adapter.safetensors").write_bytes(b"weights")

            summary = summarize_run(run_dir).to_dict()

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["checkpoint_count"], 1)
        self.assertEqual(summary["samples"][0]["name"], "step-0001.png")
        self.assertEqual(summary["samples"][0]["relative_path"], "samples/step-0001.png")
        self.assertTrue(summary["samples"][0]["is_image"])
        self.assertEqual(summary["checkpoints"][0]["name"], "adapter.safetensors")
        self.assertEqual(summary["checkpoints"][0]["relative_path"], "checkpoints/adapter.safetensors")
        self.assertFalse(summary["checkpoints"][0]["is_image"])

    def test_build_settings_summary_reports_runtime_paths_and_safety_defaults(self) -> None:
        summary = build_settings_summary(PROJECT_ROOT, env={"HF_TOKEN": "secret-token"})

        self.assertEqual(summary["backend"]["preferred"], "WSL2 Ubuntu")
        self.assertTrue(summary["hf_token"]["present"])
        self.assertNotIn("secret-token", json.dumps(summary))
        self.assertEqual(summary["runner_policy"]["max_active_jobs"], 1)
        self.assertEqual(summary["runner_policy"]["queue_order"], "fifo")

        paths = {item["id"]: item for item in summary["paths"]}
        self.assertTrue(paths["project_root"]["path"].endswith("D-opsd-T8-Tranier"))
        self.assertTrue(paths["wsl_venv"]["path"].endswith("trainer-data/venvs/dopsd"))
        self.assertTrue(paths["hf_home"]["path"].endswith("trainer-data/hf-home"))
        self.assertTrue(paths["jobs_ledger"]["path"].endswith("trainer-data/jobs/jobs.json"))

        defaults = {item["id"]: item for item in summary["safety_defaults"]}
        self.assertEqual(defaults["launcher"]["value"], "python")
        self.assertEqual(defaults["low_vram"]["value"], True)
        self.assertEqual(defaults["resolution_scale"]["value"], "profile-specific")
        self.assertEqual(defaults["use_8bit_adam"]["value"], True)
        self.assertEqual(defaults["save_samples"]["value"], True)
        self.assertEqual(defaults["save_checkpoints"]["value"], True)

    def test_production_profiles_define_verified_16gb_defaults(self) -> None:
        profiles_path = PROJECT_ROOT / "trainer_runtime" / "dopsd_trainer" / "profiles.py"
        self.assertTrue(profiles_path.exists(), "production profile registry should exist")

        spec = importlib.util.spec_from_file_location("dopsd_profiles_for_test", profiles_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        profiles = {profile["recipe_id"]: profile for profile in module.list_production_profiles()}
        self.assertEqual(set(profiles), {"flux2-klein-identity", "flux2-klein-editing", "z-image-turbo-vlm"})

        identity = profiles["flux2-klein-identity"]
        self.assertEqual(identity["tier"], "recommended_16gb")
        self.assertEqual(identity["resolution_scale"], "0.625")
        self.assertEqual(identity["sample_resolution_scale"], "0.5")
        self.assertEqual(identity["max_train_steps"], 5)
        self.assertEqual(identity["sample_steps"], 5)
        self.assertEqual(identity["checkpoint_steps"], 5)
        self.assertTrue(identity["save_samples"])
        self.assertTrue(identity["save_checkpoints"])
        self.assertFalse(identity["block_offload"])
        self.assertIn("flux2_identity_res0625_artifacts_scale05_5step_20260623045623", identity["evidence"])

        editing = profiles["flux2-klein-editing"]
        self.assertEqual(editing["resolution_scale"], "0.5625")
        self.assertEqual(editing["sample_resolution_scale"], "0.5")
        self.assertEqual(editing["max_train_steps"], 5)
        self.assertEqual(editing["sample_steps"], 5)
        self.assertEqual(editing["checkpoint_steps"], 5)
        self.assertIn("flux2_editing_res05625_artifacts_scale05_5step", " ".join(editing["evidence"]))

        zimage = profiles["z-image-turbo-vlm"]
        self.assertEqual(zimage["resolution_scale"], "0.5")
        self.assertIsNone(zimage["sample_resolution_scale"])
        self.assertEqual(zimage["max_train_steps"], 2)
        self.assertEqual(zimage["sample_steps"], 2)
        self.assertEqual(zimage["checkpoint_steps"], 2)
        self.assertIn("zimage_style_res05_artifacts_2step_202606221528", zimage["evidence"])

    def test_settings_summary_exposes_production_profiles(self) -> None:
        summary = build_settings_summary(PROJECT_ROOT, env={})

        self.assertIn("production_profiles", summary)
        profiles = {profile["recipe_id"]: profile for profile in summary["production_profiles"]}
        self.assertEqual(set(profiles), {"flux2-klein-identity", "flux2-klein-editing", "z-image-turbo-vlm"})
        self.assertEqual(profiles["flux2-klein-identity"]["resolution_scale"], "0.625")
        self.assertEqual(profiles["flux2-klein-editing"]["sample_resolution_scale"], "0.5")
        self.assertEqual(profiles["z-image-turbo-vlm"]["max_train_steps"], 2)

    def test_ui_uses_recommended_production_profiles_for_new_job_defaults(self) -> None:
        recipes_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "recipes.ts"
        wizard_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "NewJobWizard.tsx"
        i18n_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx"
        jobs_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"

        recipes_source = recipes_path.read_text(encoding="utf-8")
        wizard_source = wizard_path.read_text(encoding="utf-8")
        i18n_source = i18n_path.read_text(encoding="utf-8")
        jobs_source = jobs_path.read_text(encoding="utf-8")

        self.assertIn("productionProfile", recipes_source)
        self.assertIn("resolutionScale: '0.625'", recipes_source)
        self.assertIn("resolutionScale: '0.5625'", recipes_source)
        self.assertIn("resolutionScale: '0.5'", recipes_source)
        self.assertIn("sampleResolutionScale: '0.5'", recipes_source)

        self.assertIn("recommended16gbProfile", wizard_source)
        self.assertIn("trainingScale", wizard_source)
        self.assertIn("sampleScale", wizard_source)
        self.assertIn("trainingParameters", wizard_source)
        self.assertIn("Recommended 16GB starter", i18n_source)
        self.assertIn("推荐 16GB 起步配置", i18n_source)
        self.assertIn("Training scale", i18n_source)
        self.assertIn("Sample scale", i18n_source)
        self.assertIn("Training parameters", i18n_source)
        self.assertNotIn("Skip smoke samples", wizard_source)
        self.assertNotIn("Smoke resolution", wizard_source)

        self.assertIn("productionProfileForRecipe", jobs_source)
        self.assertIn("profile.runnerScript", jobs_source)
        self.assertIn("defaultTrainingValues", jobs_source)
        self.assertIn("RESOLUTION_SCALE=${values.resolutionScale}", jobs_source)
        self.assertIn("MAX_TRAIN_STEPS=${values.maxTrainSteps}", jobs_source)
        self.assertIn("SAVE_SAMPLES=${values.saveSamples ? '1' : '0'}", jobs_source)
        self.assertIn("SAVE_CHECKPOINTS=${values.saveCheckpoints ? '1' : '0'}", jobs_source)
        self.assertIn("SAMPLE_RESOLUTION_SCALE=${values.sampleResolutionScale}", jobs_source)
        self.assertIn("Start uses editable training controls", jobs_source)
        self.assertNotIn("Start runs the verified low-VRAM smoke profile", jobs_source)
        self.assertNotIn("Evidence: ${profile.verifiedRun}", jobs_source)

    def test_primary_user_ui_hides_test_verification_wording(self) -> None:
        user_ui_paths = [
            PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx",
            PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "recipes.ts",
            PROJECT_ROOT / "trainer-ui" / "src" / "components" / "NewJobWizard.tsx",
            PROJECT_ROOT / "trainer-ui" / "src" / "components" / "SettingsConsole.tsx",
            PROJECT_ROOT / "trainer-ui" / "src" / "app" / "page.tsx",
        ]
        combined_source = "\n".join(path.read_text(encoding="utf-8") for path in user_ui_paths)

        for forbidden in (
            "已验证",
            "烟测",
            "Verified 16GB profile",
            "verified16gbProfile",
            "verified16gbProfiles",
            "verifiedRun",
            "profile.evidence[0]",
        ):
            self.assertNotIn(forbidden, combined_source)

    def test_ui_shell_defaults_to_chinese_with_language_toggle_and_t8_brand(self) -> None:
        shell_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "AppShell.tsx"
        i18n_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx"
        layout_path = PROJECT_ROOT / "trainer-ui" / "src" / "app" / "layout.tsx"

        self.assertTrue(i18n_path.exists(), "UI should have a shared language provider")
        shell_source = shell_path.read_text(encoding="utf-8")
        i18n_source = i18n_path.read_text(encoding="utf-8")
        layout_source = layout_path.read_text(encoding="utf-8")

        self.assertIn("T8 D-OPSD Tranier", shell_source)
        self.assertIn("T8 D-OPSD Tranier", layout_source)
        self.assertIn("defaultLanguage: Language = 'zh'", i18n_source)
        self.assertIn("LanguageProvider", i18n_source)
        self.assertIn("useI18n", shell_source)
        self.assertIn("setLanguage(language === 'zh' ? 'en' : 'zh')", shell_source)
        self.assertIn("中文", shell_source)
        self.assertIn("EN", shell_source)
        self.assertIn("仪表盘", i18n_source)
        self.assertIn("新建训练", i18n_source)
        self.assertIn("任务", i18n_source)
        self.assertIn("数据集", i18n_source)
        self.assertIn("模型", i18n_source)
        self.assertIn("设置", i18n_source)
        self.assertIn("Dashboard", i18n_source)
        self.assertIn("New Job", i18n_source)

    def test_primary_pages_use_chinese_first_i18n_copy(self) -> None:
        i18n_source = (PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx").read_text(
            encoding="utf-8"
        )
        component_paths = [
            PROJECT_ROOT / "trainer-ui" / "src" / "components" / "JobsTable.tsx",
            PROJECT_ROOT / "trainer-ui" / "src" / "components" / "DatasetValidator.tsx",
            PROJECT_ROOT / "trainer-ui" / "src" / "components" / "ModelManager.tsx",
            PROJECT_ROOT / "trainer-ui" / "src" / "components" / "SettingsConsole.tsx",
        ]

        for component_path in component_paths:
            with self.subTest(component=component_path.name):
                source = component_path.read_text(encoding="utf-8")
                self.assertIn("useI18n", source)
                self.assertIn("t(", source)

        for chinese_copy in (
            "任务账本",
            "完整日志",
            "打开输出目录",
            "数据集验证器",
            "模型缓存",
            "自定义模型路径",
            "打开文件夹",
            "运行时设置",
            "运行时路径",
            "检测到的后端",
            "运行器策略",
            "低显存安全默认值",
            "推荐 16GB 起步配置",
        ):
            self.assertIn(chinese_copy, i18n_source)

        for english_copy in (
            "Job ledger",
            "Full logs",
            "Open output folder",
            "Dataset validator",
            "Model cache",
            "Custom model path",
            "Open folder",
            "Runtime settings",
            "Runtime paths",
            "Detected backend",
            "Runner policy",
        ):
            self.assertIn(english_copy, i18n_source)

    def test_models_ui_supports_custom_model_paths_and_open_folder(self) -> None:
        models_lib_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "models.ts"
        models_route_path = PROJECT_ROOT / "trainer-ui" / "src" / "app" / "api" / "models" / "route.ts"
        manager_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "ModelManager.tsx"
        i18n_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx"

        models_lib_source = models_lib_path.read_text(encoding="utf-8")
        models_route_source = models_route_path.read_text(encoding="utf-8")
        manager_source = manager_path.read_text(encoding="utf-8")
        i18n_source = i18n_path.read_text(encoding="utf-8")

        self.assertIn("custom-model-paths.json", models_lib_source)
        self.assertIn("readCustomModelPaths", models_lib_source)
        self.assertIn("saveCustomModelPath", models_lib_source)
        self.assertIn("openModelFolder", models_lib_source)
        self.assertIn("explorer.exe", models_lib_source)
        self.assertIn("action?:", models_route_source)
        self.assertIn("save-custom-path", models_route_source)
        self.assertIn("open-folder", models_route_source)
        self.assertIn("customPaths", manager_source)
        self.assertIn("customPathInputs", manager_source)
        self.assertIn("handleSaveCustomPath", manager_source)
        self.assertIn("handleOpenModelFolder", manager_source)
        self.assertIn("自定义模型路径", i18n_source)
        self.assertIn("打开文件夹", i18n_source)
        self.assertIn("Custom model path", i18n_source)
        self.assertIn("Open folder", i18n_source)

    def test_production_profile_checker_is_part_of_release_readiness(self) -> None:
        checker_path = PROJECT_ROOT / "scripts" / "check_production_profiles.ps1"
        release_path = PROJECT_ROOT / "scripts" / "check_release_readiness.ps1"

        self.assertTrue(checker_path.exists(), "production profile checker should exist")
        checker_source = checker_path.read_text(encoding="utf-8")
        release_source = release_path.read_text(encoding="utf-8")

        self.assertIn("profiles", checker_source)
        self.assertIn("recommended_16gb", checker_source)
        self.assertIn("flux2-klein-identity", checker_source)
        self.assertIn("flux2-klein-editing", checker_source)
        self.assertIn("z-image-turbo-vlm", checker_source)
        self.assertIn("0.625", checker_source)
        self.assertIn("0.5625", checker_source)
        self.assertIn("zimage_style_res05_artifacts_2step_202606221528", checker_source)

        self.assertIn("scripts\\check_production_profiles.ps1", release_source)
        self.assertIn("Production profile contract", release_source)

    def test_windows_launcher_starts_ui_with_health_check_and_logs(self) -> None:
        cmd_path = PROJECT_ROOT / "D-OPSD-Trainer.cmd"
        ps_path = PROJECT_ROOT / "scripts" / "start_trainer.ps1"
        server_path = PROJECT_ROOT / "scripts" / "run_ui_server.ps1"

        self.assertTrue(cmd_path.exists(), "root double-click launcher should exist")
        self.assertTrue(ps_path.exists(), "PowerShell launcher script should exist")
        self.assertTrue(server_path.exists(), "UI server wrapper should exist for persistent launches")

        cmd_source = cmd_path.read_text(encoding="utf-8").lower()
        ps_source = ps_path.read_text(encoding="utf-8")
        server_source = server_path.read_text(encoding="utf-8")

        self.assertIn("scripts\\start_trainer.ps1", cmd_source)
        self.assertIn("-executionpolicy bypass", cmd_source)
        self.assertIn("-wait", cmd_source)
        self.assertIn('set "path=%path%"', cmd_source)
        self.assertIn('set "should_wait=1"', cmd_source)
        self.assertIn('findstr /i /c:"-smoketest"', cmd_source)
        self.assertIn('set "launcher_args=%launcher_args% -wait"', cmd_source)
        self.assertLess(
            cmd_source.find("pwsh.exe"),
            cmd_source.find("powershell.exe"),
            "root launcher should prefer pwsh because Windows PowerShell can fail on duplicate Path/PATH environment blocks",
        )

        self.assertIn("[int]$Port = 8675", ps_source)
        self.assertIn("[switch]$NoBrowser", ps_source)
        self.assertIn("[switch]$SmokeTest", ps_source)
        self.assertIn("[switch]$Wait", ps_source)
        self.assertIn("trainer-ui", ps_source)
        self.assertIn("trainer-data", ps_source)
        self.assertIn("launcher", ps_source)
        self.assertIn("run_ui_server.ps1", ps_source)
        self.assertIn("node_modules", ps_source)
        self.assertIn("package.json", ps_source)
        self.assertIn("127.0.0.1", ps_source)
        self.assertIn("/api/project", ps_source)
        self.assertIn("Invoke-WebRequest", ps_source)
        self.assertIn("ConvertFrom-Json", ps_source)
        self.assertIn("Test-DopsdProjectResponse", ps_source)
        self.assertIn('meta.name -ne "D-OPSD Trainer"', ps_source)
        self.assertIn("Port $Port is already in use by a different service", ps_source)
        self.assertIn("Start-Process", ps_source)
        self.assertIn("-NoNewWindow", ps_source)
        self.assertIn("launcher.pid", ps_source)
        self.assertIn("wsl.exe", ps_source)
        self.assertIn("--status", ps_source)
        self.assertNotIn("wsl -l -v", ps_source)
        self.assertIn("[System.Diagnostics.Process]::GetCurrentProcess().Path", ps_source)
        self.assertIn("Stop-LauncherProcessTree", ps_source)
        self.assertIn("Wait-Process", ps_source)

        self.assertIn("npm", server_source)
        self.assertIn("run", server_source)
        self.assertIn("dev", server_source)
        self.assertIn("--prefix", server_source)
        self.assertIn("--hostname", server_source)
        self.assertIn("127.0.0.1", server_source)
        self.assertIn("--port", server_source)

    def test_wsl_trainer_setup_script_installs_training_dependencies(self) -> None:
        script_path = PROJECT_ROOT / "scripts" / "setup_wsl_trainer.ps1"

        self.assertTrue(script_path.exists(), "WSL trainer setup script should exist for fresh machines")
        source = script_path.read_text(encoding="utf-8")

        for token in (
            "[string]$Distro = \"Ubuntu-22.04\"",
            "requirements-trainer.txt",
            "dopsd_wsl_env.sh",
            "wslpath",
            "PROJECT_ROOT=",
            "python3 -m venv",
            "python -m pip install --upgrade pip wheel setuptools",
            "python -m pip install -r requirements-trainer.txt",
            "python -m pip check",
            "scripts/check_runtime.py probe",
            "scripts/check_runtime.py settings",
        ):
            self.assertIn(token, source)

    def test_ui_smoke_script_checks_key_pages_without_browser_runtime(self) -> None:
        script_path = PROJECT_ROOT / "scripts" / "check_ui_smoke.ps1"

        self.assertTrue(script_path.exists(), "UI smoke checker should exist for environments where Browser automation is unavailable")
        source = script_path.read_text(encoding="utf-8")

        self.assertIn("[string]$BaseUrl = \"http://127.0.0.1:8675\"", source)
        self.assertIn("/api/project", source)
        self.assertIn("/api/models", source)
        self.assertIn("customPaths", source)
        self.assertIn("/api/jobs", source)
        self.assertIn("/api/settings", source)
        self.assertIn("/api/datasets", source)
        self.assertIn("Invoke-WebRequest", source)
        self.assertIn("Invoke-RestMethod", source)
        self.assertIn("__next_f.push", source)
        self.assertIn("/_next/static/css/", source)
        self.assertIn("ConvertFrom-Utf8Base64", source)
        self.assertIn("VDggRC1PUFNEIFRyYW5pZXI=", source)
        self.assertIn("5paw5bu66K6t57uD", source)
        self.assertIn("5Luq6KGo55uY", source)
        self.assertIn("5Lit5paH", source)
        self.assertIn("EN", source)
        self.assertIn("6YWN5a+56aKE5qOA", source)
        self.assertIn("5Yib5bu66I2J56i/", source)
        self.assertIn("5pi+5a2Y5LiO5ZCv5Yqo", source)
        self.assertIn("5L2O5pi+5a2Y5Y246L29", source)
        self.assertIn("5o6o6I2QIDE2R0Ig6LW35q2l6YWN572u", source)
        self.assertIn("5qC35pys57yp5pS+", source)
        self.assertIn("6K6t57uD5p2D6YeN", source)
        self.assertIn("5qC35Zu+5o+Q56S66K+N", source)
        self.assertIn("5pWw5o2u6ZuG6Zi75aGe", source)
        self.assertIn("5Lu75Yqh", source)
        self.assertIn("5pWw5o2u6ZuG", source)
        self.assertIn("5qih5Z6L", source)
        self.assertIn("6K6+572u", source)
        self.assertIn("5oyB5LmF5Lu75Yqh6LSm5pys", source)
        self.assertIn("5Lu75Yqh6LSm5pys", source)
        self.assertIn("Hugging Face", source)
        self.assertIn("6L+Q6KGM5pe26Lev5b6E", source)
        self.assertIn("5pWw5o2u6ZuG6aqM6K+B5Zmo", source)
        self.assertIn("5pWw5o2u6ZuG6Lev5b6E", source)
        self.assertIn("5a+85YWl5pWw5o2u6ZuG", source)
        self.assertIn("5LiK5Lyg5Zu+5YOP", source)
        self.assertIn("5omT5qCH5paH5Lu2", source)
        self.assertIn("5aSa6YCJ5pWw5o2u6ZuG", source)
        self.assertIn("5ZCI5bm25ZCO5Yib5bu66I2J56i/", source)
        self.assertIn("5qih5Z6L57yT5a2Y", source)
        self.assertIn("6Ieq5a6a5LmJ5qih5Z6L6Lev5b6E", source)
        self.assertIn("5omT5byA5paH5Lu25aS5", source)
        self.assertIn("CssAssets", source)
        self.assertIn("PagesChecked", source)
        self.assertIn("DefaultModelsCached", source)

    def test_readme_documents_weighted_datasets_and_custom_sample_prompts(self) -> None:
        readme_source = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        for token in (
            "每个数据集设置训练权重",
            "样图提示词",
            "一行一个样图提示词",
            "训练权重为整数 0-10",
        ):
            self.assertIn(token, readme_source)

    def test_new_job_wizard_preflights_dataset_pairs_before_draft_creation(self) -> None:
        page_path = PROJECT_ROOT / "trainer-ui" / "src" / "app" / "jobs" / "new" / "page.tsx"
        wizard_path = PROJECT_ROOT / "trainer-ui" / "src" / "components" / "NewJobWizard.tsx"
        jobs_route_path = PROJECT_ROOT / "trainer-ui" / "src" / "app" / "api" / "jobs" / "route.ts"
        jobs_lib_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"

        self.assertTrue(wizard_path.exists(), "New Job page should use a client wizard with dataset preflight")
        page_source = page_path.read_text(encoding="utf-8")
        wizard_source = wizard_path.read_text(encoding="utf-8")
        i18n_source = (PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "i18n.tsx").read_text(encoding="utf-8")
        jobs_route_source = jobs_route_path.read_text(encoding="utf-8")
        jobs_lib_source = jobs_lib_path.read_text(encoding="utf-8")

        self.assertIn("NewJobWizard", page_source)
        self.assertIn("'use client'", wizard_source)
        self.assertIn("useI18n", wizard_source)
        self.assertIn("fetch('/api/datasets'", wizard_source)
        self.assertIn("t('pairPreflight')", wizard_source)
        self.assertIn("t('reference')", wizard_source)
        self.assertIn("t('target')", wizard_source)
        self.assertIn("t('datasetIssuesMustBeFixed')", wizard_source)
        self.assertIn("配对预检", i18n_source)
        self.assertIn("Pair preflight", i18n_source)
        self.assertIn("参考图", i18n_source)
        self.assertIn("Reference", i18n_source)
        self.assertIn("datasetPath", wizard_source)
        self.assertIn("t('createDraft')", wizard_source)
        self.assertIn("disabled={", wizard_source)
        self.assertIn("!selectionOk", wizard_source)
        self.assertIn("createDraftJob(payload.recipeId, payload.datasetPath, payload.datasetPaths, payload.trainingOverrides)", jobs_route_source)
        self.assertIn("datasetPath?: string", jobs_route_source)
        self.assertIn("validateDataset", jobs_lib_source)
        self.assertIn("bundledDatasets", jobs_lib_source)
        self.assertIn("datasetPath?: string", jobs_lib_source)
        self.assertIn("summary.ok", jobs_lib_source)
        self.assertIn("Dataset issues must be fixed before launch", jobs_lib_source)
        self.assertIn("datasetPath:", jobs_lib_source)
        self.assertIn("datasetRows:", jobs_lib_source)

    def test_job_start_revalidates_dataset_before_runner_launch(self) -> None:
        jobs_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "jobs.ts"
        jobs_source = jobs_path.read_text(encoding="utf-8")

        self.assertIn("async function preflightLaunchDataset(job: TrainerJob)", jobs_source)
        self.assertIn("await preflightLaunchDataset(job)", jobs_source)
        self.assertLess(
            jobs_source.find("await preflightLaunchDataset(job)", jobs_source.find("async function launchJob")),
            jobs_source.find("const runnerCommand = runnerCommandForJob(job)", jobs_source.find("async function launchJob")),
            "Dataset preflight must run before command/script creation starts a runner",
        )
        self.assertIn("Dataset issues must be fixed before launch", jobs_source)
        self.assertIn("Job dataset preflight failed before launch", jobs_source)
        self.assertIn("status: 'draft'", jobs_source[jobs_source.find("async function promoteNextQueuedJob"):])
        self.assertIn("Queue launch blocked by dataset preflight", jobs_source)

    def test_launcher_detach_smoke_checks_non_wait_background_mode(self) -> None:
        script_path = PROJECT_ROOT / "scripts" / "check_launcher_detach.ps1"

        self.assertTrue(script_path.exists(), "launcher detach checker should verify non-Wait background starts")
        source = script_path.read_text(encoding="utf-8")

        self.assertIn("[int]$Port = 18861", source)
        self.assertIn("start_trainer.ps1", source)
        self.assertIn("-NoBrowser", source)
        self.assertNotIn("'-Wait'", source)
        self.assertIn("LauncherExited", source)
        self.assertIn("HealthyAfterLauncherExit", source)
        self.assertIn("/api/project", source)
        self.assertIn("D-OPSD Trainer", source)
        self.assertIn("Get-NetTCPConnection", source)
        self.assertIn("Get-CimInstance Win32_Process", source)
        self.assertIn("Assert-NoExistingProjectUiServer", source)
        self.assertIn("Existing project UI process", source)
        self.assertIn("Stop-ProcessTree", source)
        self.assertIn("CommandLine -like", source)
        self.assertIn("Remove-Item -LiteralPath", source)

    def test_runner_recovery_stress_script_exercises_detached_monitor_recovery(self) -> None:
        script_path = PROJECT_ROOT / "scripts" / "check_runner_recovery.ps1"

        self.assertTrue(script_path.exists(), "runner recovery stress checker should exist")
        source = script_path.read_text(encoding="utf-8")

        self.assertIn("[int]$ProbeDurationSeconds = 20", source)
        self.assertIn("[string]$BaseUrl = \"http://127.0.0.1:8675\"", source)
        self.assertIn("/api/jobs", source)
        self.assertIn("$matches.Count -eq 0", source)
        self.assertIn("probe", source)
        self.assertIn("probeDurationSeconds", source)
        self.assertIn("/start", source)
        self.assertIn("/stop", source)
        self.assertIn("runnerPid", source)
        self.assertIn("linuxPidPath", source)
        self.assertIn("Test-Path", source)
        self.assertIn("Set-StaleRunnerPid", source)
        self.assertIn("jobs.json", source)
        self.assertIn("jobs.json.recovery-check.bak", source)
        self.assertIn("Copy-Item", source)
        self.assertIn("ConvertFrom-Json", source)
        self.assertIn("ConvertTo-Json", source)
        self.assertIn("UTF8Encoding", source)
        self.assertIn("WriteAllText", source)
        self.assertIn("Assert-JobsLedgerPreserved", source)
        self.assertIn("Restore-JobsLedgerBackup", source)
        self.assertIn("preJobIds", source)
        self.assertIn('notes.Contains("Runner process monitor detached")', source)
        self.assertIn("Runner process monitor detached", source)
        self.assertIn("completed", source)
        self.assertIn("finally", source)

    def test_job_queue_smoke_script_exercises_queue_clone_delete_and_ledger_recovery(self) -> None:
        script_path = PROJECT_ROOT / "scripts" / "check_job_queue_smoke.ps1"

        self.assertTrue(script_path.exists(), "job queue smoke checker should exist")
        source = script_path.read_text(encoding="utf-8")

        self.assertIn("[string]$BaseUrl = \"http://127.0.0.1:8675\"", source)
        self.assertIn("probeDurationSeconds", source)
        self.assertIn("FirstJobId", source)
        self.assertIn("SecondJobId", source)
        self.assertIn("status -eq \"queued\"", source)
        self.assertIn("Stop the job before deleting it", source)
        self.assertIn("/stop", source)
        self.assertIn("auto-promoted", source)
        self.assertIn("/clone", source)
        self.assertIn("ClonedJobId", source)
        self.assertIn("DELETE", source)
        self.assertIn("Assert-JobsLedgerPreserved", source)
        self.assertIn("finally", source)
        self.assertIn("$actualJobIds = @(Get-JobsLedgerIds)", source)
        self.assertIn("$preJobIds = @(Get-JobsLedgerIds)", source)
        self.assertNotIn("$_.ErrorDetails.Message", source)
        self.assertIn("ErrorDetails.ToString()", source)
        self.assertIn("Get-Member -Name Content", source)
        self.assertIn("Get-Member -Name GetResponseStream", source)

    def test_training_observability_smoke_script_checks_logs_stream_artifacts_and_telemetry(self) -> None:
        script_path = PROJECT_ROOT / "scripts" / "check_training_observability.ps1"

        self.assertTrue(script_path.exists(), "training observability smoke checker should exist")
        source = script_path.read_text(encoding="utf-8")

        self.assertIn("[string]$BaseUrl = \"http://127.0.0.1:8675\"", source)
        self.assertIn("[int]$ProbeDurationSeconds = 12", source)
        self.assertIn("probeDurationSeconds", source)
        self.assertIn("/api/jobs", source)
        self.assertIn("/start", source)
        self.assertIn("/stop", source)
        self.assertIn("/logs", source)
        self.assertIn("/logs/stream", source)
        self.assertIn("/artifact?path=", source)
        self.assertIn("/api/telemetry", source)
        self.assertIn("text/event-stream", source)
        self.assertIn("event: snapshot", source)
        self.assertIn("event: append", source)
        self.assertIn("event: heartbeat", source)
        self.assertIn("runner", source)
        self.assertIn("training", source)
        self.assertIn("samples/probe-sample.png", source)
        self.assertIn("samples_trajectory/probe-trajectory.png", source)
        self.assertIn("checkpoints/probe-adapter.safetensors", source)
        self.assertIn("ArtifactCounts", source)
        self.assertIn("SseEvents", source)
        self.assertIn("Assert-JobsLedgerPreserved", source)
        self.assertIn("finally", source)
        self.assertIn("$actualJobIds = @(Get-JobsLedgerIds)", source)
        self.assertIn("$preJobIds = @(Get-JobsLedgerIds)", source)

    def test_release_readiness_script_runs_core_verification_gates(self) -> None:
        script_path = PROJECT_ROOT / "scripts" / "check_release_readiness.ps1"

        self.assertTrue(script_path.exists(), "release readiness checker should exist")
        source = script_path.read_text(encoding="utf-8")

        self.assertIn("[string]$BaseUrl = \"http://127.0.0.1:8675\"", source)
        self.assertIn('$PSVersionTable.PSEdition -ne "Core"', source)
        self.assertIn("PowerShell 7", source)
        self.assertIn("python -m unittest discover -s trainer_runtime\\tests -v", source)
        self.assertIn("npm run typecheck --prefix trainer-ui", source)
        self.assertIn('-FilePath "npm.cmd"', source)
        self.assertIn("scripts\\check_ui_smoke.ps1", source)
        self.assertIn("python -m json.tool features.json", source)
        self.assertIn("python -m json.tool meta.json", source)
        self.assertIn("git -c safe.directory=E:/D-opsd-T8-Tranier diff --check", source)
        self.assertIn("PSParser", source)
        self.assertIn("run_flux2_smoke.sh", source)
        self.assertIn("run_flux2_editing_smoke.sh", source)
        self.assertIn("run_zimage_smoke.sh", source)
        self.assertIn("pgrep -af train_dopsd", source)
        self.assertIn("pgrep -af accelerate", source)
        self.assertIn("pgrep -af deepspeed", source)
        self.assertIn("nvidia-smi", source)
        self.assertIn("check_job_queue_smoke.ps1", source)
        self.assertIn("check_runner_recovery.ps1", source)
        self.assertIn("check_training_observability.ps1", source)
        self.assertIn("SkipLiveQueue", source)
        self.assertIn("SkipRunnerRecovery", source)
        self.assertIn('$previousErrorActionPreference = $ErrorActionPreference', source)
        self.assertIn('$ErrorActionPreference = "Continue"', source)
        self.assertIn('$ErrorActionPreference = $previousErrorActionPreference', source)

    def test_ui_restart_recovery_script_replays_running_jobs_after_process_restart(self) -> None:
        script_path = PROJECT_ROOT / "scripts" / "check_ui_restart_recovery.ps1"

        self.assertTrue(script_path.exists(), "UI restart recovery checker should exist")
        source = script_path.read_text(encoding="utf-8")

        self.assertIn("[int]$Port =", source)
        self.assertIn("[int]$ProbeDurationSeconds = 30", source)
        self.assertIn("run_ui_server.ps1", source)
        self.assertIn("Start-UiServer", source)
        self.assertIn("Stop-ProcessTree", source)
        self.assertIn("Stop-UiServer", source)
        self.assertIn("Get-NetTCPConnection", source)
        self.assertIn("OwningProcess", source)
        self.assertIn("netstat.exe", source)
        self.assertIn("Get-UiProcessPids", source)
        self.assertIn("Win32_Process", source)
        self.assertIn("taskkill.exe", source)
        self.assertIn("Wait-ForUiReady", source)
        self.assertIn("/api/project", source)
        self.assertIn("/api/jobs", source)
        self.assertIn("probeDurationSeconds", source)
        self.assertIn("/start", source)
        self.assertIn("/stop", source)
        self.assertIn("DELETE", source)
        self.assertIn("Wait-ForJob", source)
        self.assertIn("Wait-ForFile", source)
        self.assertIn("linuxPidHostPath", source)
        self.assertIn("status -eq \"running\"", source)
        self.assertIn("RestartedStatus", source)
        self.assertIn("CompletedStatus", source)
        self.assertIn("RunnerExitCode", source)
        self.assertIn("InitialServerProcessId", source)
        self.assertIn("RestartedServerProcessId", source)
        self.assertIn("Remove-ProbeArtifacts", source)
        self.assertIn("Assert-JobsLedgerPreserved", source)
        self.assertIn("finally", source)
        self.assertIn("$actualJobIds = @(Get-JobsLedgerIds)", source)
        self.assertIn("$preJobIds = @(Get-JobsLedgerIds)", source)

    def test_electron_packaging_contract_is_declared(self) -> None:
        package_path = PROJECT_ROOT / "trainer-ui" / "package.json"
        next_config_path = PROJECT_ROOT / "trainer-ui" / "next.config.ts"

        package = json.loads(package_path.read_text(encoding="utf-8"))
        scripts = package["scripts"]
        dev_dependencies = package["devDependencies"]

        self.assertIn("build", package)
        build_config = package["build"]
        self.assertEqual(package["main"], "electron/main.js")
        for script_name in ("build:standalone", "electron:dev", "pack:win", "dist:win"):
            self.assertIn(script_name, scripts)
        self.assertIn("next build", scripts["build:standalone"])
        self.assertIn("scripts/prepare-standalone.js", scripts["build:standalone"])
        self.assertIn("electron", scripts["electron:dev"])
        self.assertIn("electron-builder --win dir --x64", scripts["pack:win"])
        self.assertIn("electron-builder --win --x64", scripts["dist:win"])

        self.assertIn("electron", dev_dependencies)
        self.assertIn("electron-builder", dev_dependencies)
        self.assertEqual(build_config["appId"], "com.t8.dopsd.trainer")
        self.assertEqual(build_config["productName"], "T8 D-OPSD Tranier")
        self.assertEqual(build_config["directories"]["output"], "release")
        self.assertIs(build_config["asar"], False)

        packaged_files = json.dumps(build_config["files"], ensure_ascii=False)
        extra_resources = json.dumps(build_config["extraResources"], ensure_ascii=False)
        self.assertIn(".next/standalone", packaged_files)
        self.assertIn("workspace-template", extra_resources)
        for token in (
            "../scripts",
            "../trainer_runtime",
            "../z-image-turbo_self-distill-vlm",
            "../flux2-klein_self-distill-edit",
            "../flux2-klein-edit-self-distill-gt-ref",
            "../requirements-trainer.txt",
            "../meta.json",
            "../features.json",
            "../docs",
        ):
            self.assertIn(token, extra_resources)

        next_config = next_config_path.read_text(encoding="utf-8")
        self.assertRegex(next_config, r"output:\s*['\"]standalone['\"]")

    def test_electron_main_runs_standalone_next_with_writable_workspace(self) -> None:
        main_path = PROJECT_ROOT / "trainer-ui" / "electron" / "main.js"
        prepare_path = PROJECT_ROOT / "trainer-ui" / "scripts" / "prepare-standalone.js"

        self.assertTrue(main_path.exists(), "Electron main process should exist")
        self.assertTrue(prepare_path.exists(), "standalone preparation script should exist")

        main_source = main_path.read_text(encoding="utf-8")
        prepare_source = prepare_path.read_text(encoding="utf-8")

        for token in (
            "DOPSD_PROJECT_ROOT",
            "ELECTRON_RUN_AS_NODE",
            "workspace-template",
            "app.getPath('userData')",
            "/api/project",
            "--smoke-test",
            "DOPSD_ELECTRON_PORT",
            "DOPSD_ELECTRON_SMOKE_FILE",
            "T8 D-OPSD Tranier",
        ):
            self.assertIn(token, main_source)
        self.assertIn("copyTemplateIntoWorkspace", main_source)
        self.assertIn("waitForServer", main_source)
        self.assertIn("server.js", main_source)
        self.assertIn("process.resourcesPath", main_source)
        self.assertIn("appendElectronLog", main_source)
        self.assertIn("safeStreamWrite", main_source)
        self.assertIn("process.stdout?.on('error'", main_source)
        self.assertNotIn("process.stdout.write(`[next]", main_source)
        self.assertNotIn("process.stderr.write(`[next]", main_source)

        self.assertIn(".next/static", prepare_source)
        self.assertIn(".next/standalone", prepare_source)
        self.assertIn("copyRecursive", prepare_source)
        self.assertIn("server.js", prepare_source)

    def test_next_server_project_root_can_be_overridden_for_packaged_workspace(self) -> None:
        helper_path = PROJECT_ROOT / "trainer-ui" / "src" / "lib" / "project.ts"
        self.assertTrue(helper_path.exists(), "server helpers need an env-aware project root resolver")

        helper_source = helper_path.read_text(encoding="utf-8")
        self.assertIn("DOPSD_PROJECT_ROOT", helper_source)
        self.assertIn("resolveProjectRoot", helper_source)

        for relative in (
            "trainer-ui/src/lib/datasets.ts",
            "trainer-ui/src/lib/jobs.ts",
            "trainer-ui/src/lib/models.ts",
            "trainer-ui/src/lib/settings.ts",
            "trainer-ui/src/app/api/project/route.ts",
            "trainer-ui/src/app/api/system/route.ts",
        ):
            with self.subTest(relative=relative):
                source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
                self.assertIn("resolveProjectRoot", source)
                self.assertNotIn("path.resolve(process.cwd(), '..')", source)


if __name__ == "__main__":
    unittest.main()
