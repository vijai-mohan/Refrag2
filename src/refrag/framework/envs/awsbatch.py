"""AWS Batch env for Hydra applications.

This env submits jobs to AWS Batch with support for:
- Dynamic job definition registration with ECR images
- GPU and CPU instance selection
- Workspace overlay (syncs local code to S3)
- Hydra configuration overrides
- Environment variable passthrough
"""
import os
import tempfile
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any
import logging
import sys
import shlex  # added for quoting command when wrapping bootstrap

import boto3
from botocore.exceptions import ClientError
from refrag.framework.drivers import transform_overrides
from omegaconf import DictConfig
from .launcher_utils import build_host_cmd, make_paths_relative  # added shared helper import

log = logging.getLogger(__name__)


@dataclass
class AWSBatchLauncher:
    """Env for submitting jobs to AWS Batch."""

    region: str
    job_queue: str = ""  # Auto-selected based on GPU count if not specified
    job_definition_name: str = ""  # Auto-selected based on GPU count if not specified
    cpu_job_queue: str = "refrag-cpu-job-queue"
    gpu_job_queue: str = "refrag-gpu-job-queue"
    cpu_job_definition_name: str = "refrag-cpu-job-def"
    gpu_job_definition_name: str = "refrag-gpu-job-def"
    vcpus: int = 4
    memory: int = 8192
    gpus: int = 0  # legacy single-host total GPUs
    # Distributed
    num_hosts: int = 1
    gpus_per_host: int = 1

    env_passthrough: List[str] = field(default_factory=list)
    overlay: Optional[DictConfig] = None
    job_timeout_seconds: int = 86400
    retry_attempts: int = 1
    overlay_extract_root: str = "/workspace"  # target directory inside container

    def __post_init__(self):
        """Initialize AWS clients."""
        self.batch = boto3.client("batch", region_name=self.region)
        self.s3 = boto3.client("s3", region_name=self.region)
        # CloudWatch Logs client for streaming job logs
        self.logs = boto3.client("logs", region_name=self.region)
        # ECR client intentionally not created: image lifecycle and registry operations
        # are managed by Terraform and CI/CD; env uses environment overrides.

    def _get_queue_and_jobdef(self, gpus: int) -> tuple[str, str]:
        """Select appropriate queue and job definition based on GPU requirement.

        Args:
            gpus: Number of GPUs requested

        Returns:
            Tuple of (job_queue_name, job_definition_name)
        """
        # If explicitly set, use those
        if self.job_queue and self.job_definition_name:
            return self.job_queue, self.job_definition_name

        # Auto-select based on GPU count
        if gpus > 0:
            queue = self.gpu_job_queue
            jobdef = self.gpu_job_definition_name
            log.info(f"Auto-selected GPU queue: {queue}")
        else:
            queue = self.cpu_job_queue
            jobdef = self.cpu_job_definition_name
            log.info(f"Auto-selected CPU queue: {queue}")

        return queue, jobdef

    def _create_overlay(self, cfg: DictConfig) -> Optional[str]:
        """Create workspace overlay tarball and upload to S3.

        Returns:
            S3 URI of uploaded tarball, or None if overlay disabled
        """
        if not self.overlay or not self.overlay.get("enabled"):
            log.info("Overlay disabled, skipping workspace sync")
            return None

        log.info("Creating workspace overlay...")
        workspace_root = Path.cwd()

        include_patterns = self.overlay.get("include_patterns", ["**/*.py"])
        exclude_patterns = self.overlay.get("exclude_patterns", [])

        # Create tarball
        overlay_files = []
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                for pattern in include_patterns:
                    for path in workspace_root.glob(pattern):
                        if path.is_file():
                            # Check exclusions
                            should_exclude = any(
                                path.match(excl) for excl in exclude_patterns
                            )
                            if not should_exclude:
                                arcname = path.relative_to(workspace_root)
                                tar.add(path, arcname=arcname)
                                overlay_files.append(str(arcname))
                                log.debug(f"Added to overlay: {arcname}")

            log.info(f"Overlay contains {len(overlay_files)} files")

            # Upload to S3
            s3_bucket = self.overlay.get("s3_bucket")
            s3_prefix = self.overlay.get("s3_prefix", "jobs/batch")
            run_id = cfg.get("run_id", int(time.time()))
            s3_key = f"{s3_prefix}/{run_id}/workspace.tar.gz"

            log.info(f"Uploading overlay to s3://{s3_bucket}/{s3_key}")
            try:
                self.s3.upload_file(tmp.name, s3_bucket, s3_key)
                log.info("Overlay uploaded successfully")
            except ClientError as e:
                log.error(f"Failed to upload overlay: {e}")
                raise
            finally:
                os.unlink(tmp.name)

            return f"s3://{s3_bucket}/{s3_key}"

    def _get_ecr_image(self) -> str:
        """Get the ECR image to use.

        Returns:
            Image reference to use for the job (can be a short name like 'refrag:latest' or full URI).

        Notes:
            We intentionally avoid calling the ECR API here: image/repository lifecycle is managed by Terraform.
            Priority is:
              1. ENV var `ECR_IMAGE` (full image URI or short name)
              2. ENV var `ECR_REPO_NAME` (defaults to 'refrag') -> returned as '<repo>:latest'
        """
        # Allow explicit override of the full image (e.g. '123456789012.dkr.ecr.us-east-1.amazonaws.com/refrag:latest')
        explicit = os.getenv("ECR_IMAGE")
        if explicit:
            log.info(f"Using explicit ECR image from ECR_IMAGE: {explicit}")
            return explicit

        # Fall back to repo name only (short image name like 'refrag:latest')
        repo_name = os.getenv("ECR_REPO_NAME", "refrag")
        image_uri = f"{repo_name}:latest"
        log.info(f"Using ECR image: {image_uri}")
        return image_uri

    # -------------------- Job monitoring & logs --------------------
    def _describe_job(self, job_id: str) -> dict:
        try:
            resp = self.batch.describe_jobs(jobs=[job_id])
        except ClientError as e:
            raise RuntimeError(f"batch.describe_jobs failed: {e}")
        jobs = resp.get("jobs", [])
        if not jobs:
            raise RuntimeError(f"job not found: {job_id}")
        return jobs[0]

    def _extract_log_stream(self, job: dict) -> str:
        # container.logStreamName is set when the container starts; attempts may hold it later
        container = job.get("container", {}) or {}
        log_stream = container.get("logStreamName")
        if not log_stream:
            attempts = job.get("attempts", [])
            if attempts:
                log_stream = attempts[-1].get("container", {}).get("logStreamName")
        if not log_stream:
            raise RuntimeError(f"log stream not found for job {job.get('jobId')}")
        return log_stream

    def _stream_logs_for_stream(self, log_group: str, log_stream: str, next_tokens: dict, poll_interval: float = 2.0):
        """Fetch and print new log events for a single stream. Maintains nextForwardToken in next_tokens."""
        next_token = next_tokens.get(log_stream)
        try:
            kwargs = {
                "logGroupName": log_group,
                "logStreamName": log_stream,
                "startFromHead": True,
            }
            if next_token:
                kwargs["nextToken"] = next_token
            resp = self.logs.get_log_events(**kwargs)
        except ClientError as e:
            # Logs may not be available yet; caller will handle retrying
            raise RuntimeError(f"logs.get_log_events failed: {e}")

        events = resp.get("events", [])
        for ev in events:
            ts = ev.get("timestamp", 0)
            msg = ev.get("message", "")
            # Print timestamped message (human-readable timestamp optional)
            print(f"[{log_stream}] {ts}: {msg}")

        # update token for subsequent incremental reads
        new_token = resp.get("nextForwardToken")
        if new_token:
            next_tokens[log_stream] = new_token

    def track_job_and_stream_logs(self, job_id: str, log_group: str = "/aws/batch/job", poll_interval: float = 2.0) -> str:
        """Block until job reaches terminal state while streaming logs.

        Returns final job status (e.g. 'SUCCEEDED' or 'FAILED').
        """
        terminal_states = {"SUCCEEDED", "FAILED"}
        last_status = None
        next_tokens: dict = {}
        log.info(f"Tracking job {job_id} and streaming logs from {log_group}")

        while True:
            job = self._describe_job(job_id)
            status = job.get("status")
            if status != last_status:
                log.info(f"Job {job_id} status: {status}")
                last_status = status

            # attempt to stream logs for the current job/attempt
            try:
                log_stream = self._extract_log_stream(job)
                # stream any new events for this stream
                try:
                    self._stream_logs_for_stream(log_group, log_stream, next_tokens)
                except RuntimeError as e:
                    log.debug(f"No logs yet for job {job_id}: {e}")
            except RuntimeError:
                # log stream not available yet; ignore and continue polling
                pass

            if status in terminal_states:
                log.info(f"Job {job_id} reached terminal state: {status}")
                return status

            time.sleep(poll_interval)

    # -------------------- end job monitoring & logs --------------------

    def run(self, cfg: DictConfig):
        """Derive internal command from original host invocation (sys.argv) using shared helper,
        optionally download & extract overlay, transform env selector, and submit job while streaming logs."""
        # Build host command via shared helper (normalizes script path)
        host_cmd: List[str] = build_host_cmd(sys.argv, python_bin="python3")
        # Transform env overrides (not injecting if absent)
        base_cmd = transform_overrides(cfg, host_cmd, group_key="env", target_value="local")

        # Ensure output paths are relative to project root so bootstrap and container commands are correct
        base_cmd = make_paths_relative(base_cmd)

        overlay_uri = self._create_overlay(cfg)
        if overlay_uri:
            quoted_original = " ".join(shlex.quote(tok) for tok in base_cmd)
            bootstrap_parts = [
                "set -e",
                f"echo 'Overlay enabled: $OVERLAY_S3_URI'",
                f"aws s3 cp \"$OVERLAY_S3_URI\" /tmp/workspace.tar.gz",
                f"mkdir -p {shlex.quote(self.overlay_extract_root)}",
                f"tar -xzf /tmp/workspace.tar.gz -C {shlex.quote(self.overlay_extract_root)}",
                f"cd {shlex.quote(self.overlay_extract_root)}",
                quoted_original,
            ]
            full_shell = "; ".join(bootstrap_parts)
            cmd: List[str] = ["bash", "-lc", full_shell]
        else:
            cmd = base_cmd

        gpu_count = self.gpus
        num_hosts = self.num_hosts
        gpus_per_host = self.gpus_per_host
        for arg in base_cmd:
            if isinstance(arg, str):
                if arg.startswith("gpus="):
                    try:
                        gpu_count = int(arg.split("=")[1])
                    except Exception:
                        pass
                elif arg.startswith("num_hosts="):
                    try:
                        num_hosts = int(arg.split("=")[1])
                    except Exception:
                        pass
                elif arg.startswith("gpus_per_host="):
                    try:
                        gpus_per_host = int(arg.split("=")[1])
                    except Exception:
                        pass

        self.num_hosts = num_hosts
        self.gpus_per_host = gpus_per_host
        self.gpus = gpu_count if num_hosts <= 1 else gpus_per_host * num_hosts

        job_queue, base_job_def = self._get_queue_and_jobdef(self.gpus)
        job_def = self.job_definition_name or base_job_def

        # New job name pattern: app_name-run_name-YYYYMMDD-HHMMSS
        from datetime import datetime, timezone
        dt = datetime.now(timezone.utc)
        try:
            app_name_val = cfg.app_name
        except AttributeError:
            app_name_val = "app"
        try:
            run_name_val = cfg.run_name
        except AttributeError:
            run_name_val = "run"
        job_name = f"{app_name_val}-{run_name_val}-{dt.strftime('%Y%m%d')}-{dt.strftime('%H%M%S')}"

        log.info("=" * 80)
        log.info("AWS Batch Env (sys.argv derived)")
        log.info("Submitting job: %s (hosts=%d gpus_total=%d overlay=%s)", job_name, num_hosts, self.gpus, overlay_uri or "none")

        try:
            env_list = []
            if overlay_uri:
                env_list.append({"name": "OVERLAY_S3_URI", "value": overlay_uri})
            for key in self.env_passthrough:
                val = os.environ.get(key)
                if val is not None:
                    env_list.append({"name": key, "value": val})

            if num_hosts <= 1:
                container_overrides: dict[str, Any] = {"command": cmd}
                if env_list:
                    container_overrides["environment"] = env_list
                if self.gpus > 0:
                    container_overrides["resourceRequirements"] = [{"type": "GPU", "value": str(self.gpus)}]
                response = self.batch.submit_job(
                    jobName=job_name,
                    jobQueue=job_queue,
                    jobDefinition=job_def,
                    containerOverrides=container_overrides,
                )
            else:
                node_container_overrides: dict[str, Any] = {"command": cmd}
                if env_list:
                    node_container_overrides["environment"] = env_list
                if gpus_per_host > 0:
                    node_container_overrides["resourceRequirements"] = [{"type": "GPU", "value": str(int(gpus_per_host))}]
                node_overrides: Any = {
                    "numNodes": int(num_hosts),
                    "nodePropertyOverrides": [
                        {
                            "targetNodes": f"0:{int(num_hosts)-1}",
                            "containerOverrides": node_container_overrides,
                        }
                    ],
                }
                response = self.batch.submit_job(
                    jobName=job_name,
                    jobQueue=job_queue,
                    jobDefinition=job_def,
                    nodeOverrides=node_overrides,
                )

            job_id = response["jobId"]
            log.info(
                "Submitted AWS Batch job %s (queue=%s jobDef=%s hosts=%d gpus_per_host=%d overlay=%s)",
                job_id, job_queue, job_def, num_hosts, gpus_per_host, overlay_uri or "none",
            )
            final_status = self.track_job_and_stream_logs(job_id)
            if final_status != "SUCCEEDED":
                raise RuntimeError(f"Job {job_id} finished with status: {final_status}")
            return {"status": final_status, "job_id": job_id, "overlay": overlay_uri, "job_name": job_name}
        except ClientError as e:
            raise RuntimeError(f"Failed to submit AWS Batch job: {e}")
