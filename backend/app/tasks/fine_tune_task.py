"""Celery task that orchestrates the full fine-tuning pipeline.

Supports three modes:
  - Vertex AI (default): exports data → submits Vertex AI CustomJob → polls → downloads model
  - GCP VM mode: exports data → creates GPU VM → trains remotely → downloads model
  - Local mode: runs training on the same machine (requires GPU)
"""

import logging
import time
from datetime import datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models.model_version import ModelVersion
from app.models.training_run import TrainingRun
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# Sync engine for Celery workers (they run synchronously)
_sync_url = settings.database_url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg2")
sync_engine = create_engine(_sync_url)
SyncSession = sessionmaker(sync_engine)


def _post_training_coaching(self, training_run_id: int, db):
    """Run the Gemini coaching agent after training completes (non-fatal)."""
    try:
        run = db.execute(
            select(TrainingRun).where(TrainingRun.id == training_run_id)
        ).scalar_one()
        run.coaching_status = "generating"
        db.commit()

        logger.info(f"[Run {training_run_id}] Generating coaching report...")
        self.update_state(state="PROGRESS", meta={"step": "coaching_report"})

        from app.services.coaching_agent import generate_coaching_report_sync
        report = generate_coaching_report_sync(training_run_id, db)

        run.coaching_status = "completed"
        db.commit()

        logger.info(
            f"[Run {training_run_id}] Coaching report #{report.id} — "
            f"round {report.next_round_number}, {report.texts_generated} texts"
        )
        return report.id
    except Exception as e:
        logger.error(f"[Run {training_run_id}] Coaching report failed: {e}", exc_info=True)
        try:
            run = db.execute(
                select(TrainingRun).where(TrainingRun.id == training_run_id)
            ).scalar_one()
            run.coaching_status = "failed"
            db.commit()
        except Exception:
            pass
        return None


@celery_app.task(bind=True, name="fine_tune")
def fine_tune_task(self, training_run_id: int):
    """Route to Vertex AI, GCP VM, or local training based on settings."""
    if settings.vertex_training_enabled:
        return _run_vertex_training(self, training_run_id)
    elif settings.gcp_training_enabled:
        return _run_gcp_training(self, training_run_id)
    else:
        return _run_local_training(self, training_run_id)


def _run_vertex_training(self, training_run_id: int):
    """
    Vertex AI pipeline:
      1. Export training data from DB
      2. Upload data + code to GCS
      3. Submit Vertex AI CustomJob
      4. Poll job until completion
      5. Download model + register
      6. Run coaching agent
    """
    from training.config import TRAINING_CONFIG
    from app.services.gcp_training_service import (
        check_training_results,
        download_model,
        export_training_data,
        upload_code_to_gcs,
        upload_training_data,
    )
    from app.services.vertex_training_service import (
        poll_vertex_job,
        submit_vertex_training_job,
    )

    with SyncSession() as db:
        run = db.execute(
            select(TrainingRun).where(TrainingRun.id == training_run_id)
        ).scalar_one()

        run.status = "running"
        run.started_at = datetime.utcnow()
        run.celery_task_id = self.request.id
        db.commit()

        try:
            config = {
                "num_epochs": run.num_epochs,
                "lora_rank": run.lora_rank,
                "learning_rate": run.learning_rate,
                "tts_enabled": TRAINING_CONFIG.get("tts_enabled", False),
                "tts_num_synthetic": TRAINING_CONFIG.get("tts_num_synthetic", 500),
                "tts_reference_clips": TRAINING_CONFIG.get("tts_reference_clips", 5),
                "real_sample_weight": TRAINING_CONFIG.get("real_sample_weight", 8),
                "lora_encoder_layers": TRAINING_CONFIG.get("lora_encoder_layers", []),
                "spec_augment_freq_mask": TRAINING_CONFIG.get("spec_augment_freq_mask"),
                "spec_augment_time_mask": TRAINING_CONFIG.get("spec_augment_time_mask"),
            }

            # 1. Export data
            logger.info(f"[Run {training_run_id}] Exporting training data...")
            self.update_state(state="PROGRESS", meta={"step": "exporting_data"})
            export_dir, sample_count = export_training_data(training_run_id, config)
            logger.info(f"[Run {training_run_id}] Exported {sample_count} samples")

            if sample_count < 10:
                raise ValueError(f"Not enough samples: {sample_count} (need >= 10)")

            # 2. Upload to GCS
            logger.info(f"[Run {training_run_id}] Uploading to GCS...")
            self.update_state(state="PROGRESS", meta={"step": "uploading_data"})
            upload_training_data(export_dir, training_run_id)
            upload_code_to_gcs()

            # 3. Submit Vertex AI job
            logger.info(f"[Run {training_run_id}] Submitting Vertex AI job...")
            self.update_state(state="PROGRESS", meta={"step": "submitting_vertex_job"})
            job = submit_vertex_training_job(training_run_id)
            run.error_message = f"Vertex AI: {job.resource_name}"
            db.commit()

            # 4. Poll for completion
            logger.info(f"[Run {training_run_id}] Polling Vertex AI job...")
            self.update_state(state="PROGRESS", meta={"step": "training"})
            job_state = poll_vertex_job(job)

            if job_state != "SUCCEEDED":
                raise RuntimeError(f"Vertex AI job {job_state}")

            # Check results.json on GCS
            results = check_training_results(training_run_id)
            if not results or results.get("status") != "completed":
                raise RuntimeError(f"Training completed but no valid results.json found")

            # 5. Download model
            logger.info(f"[Run {training_run_id}] Downloading trained model...")
            self.update_state(state="PROGRESS", meta={"step": "downloading_model"})
            model_path = download_model(training_run_id)

            # 6. Register model version
            version_tag = f"v{training_run_id}"
            model_version = ModelVersion(
                version_tag=version_tag,
                display_name=f"Fine-tuned (run #{training_run_id}, {run.num_samples} samples, Vertex AI)",
                base_model_name=settings.hf_model_id,
                model_path=model_path,
                is_active=False,
                is_base=False,
                eval_wer=results.get("eval_wer"),
                num_training_samples=run.num_samples,
            )
            db.add(model_version)
            db.flush()

            run.status = "completed"
            run.result_model_version_id = model_version.id
            run.eval_wer = results.get("eval_wer")
            run.train_wer = results.get("train_wer")
            run.training_loss = results.get("train_loss")
            run.error_message = None
            run.completed_at = datetime.utcnow()
            db.commit()

            logger.info(f"[Run {training_run_id}] Completed! Model version: {model_version.id}")

            # Post-training: coaching agent
            coaching_id = _post_training_coaching(self, training_run_id, db)

            return {
                "status": "completed",
                "model_version_id": model_version.id,
                "coaching_report_id": coaching_id,
            }

        except Exception as e:
            logger.error(f"[Run {training_run_id}] Vertex AI training failed: {e}", exc_info=True)
            run.status = "failed"
            run.error_message = str(e)[:500]
            run.completed_at = datetime.utcnow()
            db.commit()
            raise


def _run_gcp_training(self, training_run_id: int):
    """
    GCP pipeline:
      1. Export training data from DB
      2. Upload data + code to GCS
      3. Create spot GPU VM
      4. Poll for completion
      5. Download model + register
      6. Delete VM
    """
    from training.config import TRAINING_CONFIG
    from app.services.gcp_training_service import (
        check_training_results,
        check_vm_status,
        create_training_vm,
        delete_vm,
        download_model,
        export_training_data,
        upload_code_to_gcs,
        upload_training_data,
    )

    with SyncSession() as db:
        run = db.execute(
            select(TrainingRun).where(TrainingRun.id == training_run_id)
        ).scalar_one()

        run.status = "running"
        run.started_at = datetime.utcnow()
        run.celery_task_id = self.request.id
        db.commit()

        vm_name = None

        try:
            config = {
                "num_epochs": run.num_epochs,
                "lora_rank": run.lora_rank,
                "learning_rate": run.learning_rate,
                "tts_enabled": TRAINING_CONFIG.get("tts_enabled", False),
                "tts_num_synthetic": TRAINING_CONFIG.get("tts_num_synthetic", 500),
                "tts_reference_clips": TRAINING_CONFIG.get("tts_reference_clips", 5),
                "real_sample_weight": TRAINING_CONFIG.get("real_sample_weight", 8),
                "lora_encoder_layers": TRAINING_CONFIG.get("lora_encoder_layers", []),
                "spec_augment_freq_mask": TRAINING_CONFIG.get("spec_augment_freq_mask"),
                "spec_augment_time_mask": TRAINING_CONFIG.get("spec_augment_time_mask"),
            }

            # 1. Export data
            logger.info(f"[Run {training_run_id}] Exporting training data...")
            self.update_state(state="PROGRESS", meta={"step": "exporting_data"})
            export_dir, sample_count = export_training_data(training_run_id, config)
            logger.info(f"[Run {training_run_id}] Exported {sample_count} samples")

            if sample_count < 10:
                raise ValueError(f"Not enough samples: {sample_count} (need >= 10)")

            # 2. Upload to GCS
            logger.info(f"[Run {training_run_id}] Uploading to GCS...")
            self.update_state(state="PROGRESS", meta={"step": "uploading_data"})
            job_dir = upload_training_data(export_dir, training_run_id)

            # Upload training code too
            upload_code_to_gcs()

            # 3. Create VM
            logger.info(f"[Run {training_run_id}] Creating GPU VM...")
            self.update_state(state="PROGRESS", meta={"step": "creating_vm"})
            vm_name = create_training_vm(training_run_id)
            run.error_message = f"VM: {vm_name}"  # Track VM name for debugging
            db.commit()

            # 4. Poll for completion (check every 60s, timeout after 4 hours)
            logger.info(f"[Run {training_run_id}] Training on VM {vm_name}...")
            self.update_state(state="PROGRESS", meta={"step": "training", "vm": vm_name})

            max_wait = 4 * 60 * 60  # 4 hours
            poll_interval = 60  # 60 seconds
            waited = 0

            while waited < max_wait:
                time.sleep(poll_interval)
                waited += poll_interval

                # Check if results are available
                results = check_training_results(training_run_id)
                if results:
                    logger.info(f"[Run {training_run_id}] Training results found: {results}")
                    break

                # Check VM status
                vm_status = check_vm_status(vm_name)
                if vm_status in ("TERMINATED", "STOPPED", "NOT_FOUND"):
                    # VM stopped — check results one more time
                    results = check_training_results(training_run_id)
                    if results:
                        break
                    raise RuntimeError(f"VM {vm_name} stopped/terminated without producing results (status: {vm_status})")

                if waited % 300 == 0:  # Log every 5 min
                    logger.info(f"[Run {training_run_id}] Still training... ({waited}s elapsed, VM: {vm_status})")

            if not results:
                raise TimeoutError(f"Training timed out after {max_wait}s")

            if results.get("status") != "completed":
                raise RuntimeError(f"Training failed: {results.get('error', 'unknown')}")

            # 5. Download model
            logger.info(f"[Run {training_run_id}] Downloading trained model...")
            self.update_state(state="PROGRESS", meta={"step": "downloading_model"})
            model_path = download_model(training_run_id)

            # 6. Register model version + WER gate
            new_wer = results.get("eval_wer")

            # Check current active model's WER
            current_active = db.execute(
                select(ModelVersion).where(ModelVersion.is_active == True)
            ).scalar_one_or_none()
            current_wer = current_active.eval_wer if current_active else None

            # Activate only if new model is better (lower WER)
            should_activate = False
            if current_wer is None or new_wer is None:
                should_activate = True  # No baseline to compare — activate by default
                logger.info(f"[Run {training_run_id}] No WER baseline — activating new model")
            elif new_wer < current_wer:
                should_activate = True
                improvement = current_wer - new_wer
                logger.info(f"[Run {training_run_id}] WER improved: {current_wer:.4f} → {new_wer:.4f} (Δ{improvement:.4f}) — activating!")
            else:
                logger.warning(f"[Run {training_run_id}] WER did NOT improve: {current_wer:.4f} → {new_wer:.4f} — keeping current model")

            version_tag = f"v{training_run_id}"
            model_version = ModelVersion(
                version_tag=version_tag,
                display_name=f"Fine-tuned (run #{training_run_id}, {run.num_samples} samples, GCP)",
                base_model_name=settings.hf_model_id,
                model_path=model_path,
                is_active=False,  # Set below if passing gate
                is_base=False,
                eval_wer=new_wer,
                eval_wer_improvement=(current_wer - new_wer) if current_wer and new_wer else None,
                num_training_samples=run.num_samples,
            )
            db.add(model_version)
            db.flush()

            if should_activate:
                # Deactivate current model
                if current_active:
                    current_active.is_active = False
                model_version.is_active = True
                logger.info(f"[Run {training_run_id}] Model v{model_version.id} activated (WER: {new_wer})")
            else:
                logger.info(f"[Run {training_run_id}] Model v{model_version.id} saved but NOT activated (WER gate)")

            # Update training run
            run.status = "completed"
            run.result_model_version_id = model_version.id
            run.eval_wer = new_wer
            run.train_wer = results.get("train_wer")
            run.training_loss = results.get("train_loss")
            run.error_message = None if should_activate else f"WER gate: {new_wer:.4f} >= current {current_wer:.4f}"
            run.completed_at = datetime.utcnow()
            db.commit()

            logger.info(f"[Run {training_run_id}] Completed! Model v{model_version.id}, active={should_activate}")

            # Post-training: coaching agent generates report + next round texts
            coaching_id = _post_training_coaching(self, training_run_id, db)

            return {
                "status": "completed",
                "model_version_id": model_version.id,
                "coaching_report_id": coaching_id,
            }

        except Exception as e:
            logger.error(f"[Run {training_run_id}] GCP training failed: {e}", exc_info=True)
            run.status = "failed"
            run.error_message = str(e)[:500]
            run.completed_at = datetime.utcnow()
            db.commit()
            raise

        finally:
            # Always clean up VM
            if vm_name:
                try:
                    logger.info(f"[Run {training_run_id}] Cleaning up VM {vm_name}...")
                    delete_vm(vm_name)
                except Exception:
                    logger.warning(f"Failed to delete VM {vm_name}", exc_info=True)


def _run_local_training(self, training_run_id: int):
    """
    Local pipeline (original): build dataset → LoRA train → merge → CT2 → register.
    Requires a local GPU.
    """
    with SyncSession() as db:
        run = db.execute(
            select(TrainingRun).where(TrainingRun.id == training_run_id)
        ).scalar_one()

        run.status = "running"
        run.started_at = datetime.utcnow()
        run.celery_task_id = self.request.id
        db.commit()

        try:
            # 1. Build dataset
            from training.dataset import build_training_dataset

            dataset = build_training_dataset(db, settings.audio_storage_path)
            logger.info(f"Built dataset with {len(dataset)} samples")

            # 2. Split train/eval
            split = dataset.train_test_split(test_size=0.1, seed=42)
            train_ds = split["train"]
            eval_ds = split["test"]

            # 3. Run LoRA training
            from training.trainer import run_training

            output_dir = str(
                settings.model_storage_path + f"/training_run_{training_run_id}"
            )
            result = run_training(
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                output_dir=output_dir,
                config={
                    "num_train_epochs": run.num_epochs,
                    "lora_r": run.lora_rank,
                    "learning_rate": run.learning_rate,
                },
            )
            logger.info(f"Training complete. Result: {result}")

            # 4. Merge LoRA + convert to CT2
            from training.export import merge_and_export

            version_tag = f"v{training_run_id}"
            model_dir = str(settings.model_storage_path + f"/{version_tag}")
            ct2_path = merge_and_export(result["adapter_path"], model_dir)
            logger.info(f"Model exported to {ct2_path}")

            # 5. Register new model version
            model_version = ModelVersion(
                version_tag=version_tag,
                display_name=f"Fine-tuned (run #{training_run_id}, {run.num_samples} samples)",
                base_model_name=settings.hf_model_id,
                model_path=ct2_path,
                adapter_path=result.get("adapter_path"),
                is_active=False,
                is_base=False,
                eval_wer=result.get("eval_wer"),
                num_training_samples=run.num_samples,
            )
            db.add(model_version)
            db.flush()

            # 6. Update training run
            run.status = "completed"
            run.result_model_version_id = model_version.id
            run.eval_wer = result.get("eval_wer")
            run.train_wer = result.get("train_wer")
            run.training_loss = result.get("train_loss")
            run.completed_at = datetime.utcnow()
            db.commit()

            logger.info(f"Training run {training_run_id} completed successfully")

            # Post-training: coaching agent generates report + next round texts
            coaching_id = _post_training_coaching(self, training_run_id, db)

            return {
                "status": "completed",
                "model_version_id": model_version.id,
                "coaching_report_id": coaching_id,
            }

        except Exception as e:
            logger.error(f"Training run {training_run_id} failed: {e}", exc_info=True)
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.utcnow()
            db.commit()
            raise
