#!/usr/bin/env python3
"""
GitLab Pipeline Monitor Agent

This agent monitors nightly GitLab pipeline runs and identifies new failures
between the last two scheduled pipeline runs.

Usage:
    python gitlab_pipeline_monitor.py [--token YOUR_TOKEN] [--verbose]

Requirements:
    pip install requests python-gitlab

Environment Variables:
    GITLAB_PRIVATE_TOKEN: Your GitLab personal access token
    GITLAB_URL: GitLab instance URL (default: https://gitlab-master.nvidia.com)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


class JobStatus(Enum):
    """GitLab job status values"""

    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELED = "canceled"
    SKIPPED = "skipped"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


@dataclass
class Job:
    """Represents a GitLab CI job"""

    id: int
    name: str
    status: str
    stage: str
    web_url: str
    failure_reason: Optional[str] = None
    allow_failure: bool = False

    @property
    def is_failed(self) -> bool:
        return self.status == JobStatus.FAILED.value and not self.allow_failure

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Job):
            return self.name == other.name
        return False


@dataclass
class Pipeline:
    """Represents a GitLab CI pipeline"""

    id: int
    ref: str
    status: str
    source: str
    created_at: str
    web_url: str
    jobs: List[Job] = field(default_factory=list)

    @property
    def failed_jobs(self) -> List[Job]:
        return [j for j in self.jobs if j.is_failed]

    @property
    def failed_job_names(self) -> set:
        return {j.name for j in self.failed_jobs}


class GitLabPipelineMonitor:
    """
    Agent to monitor GitLab pipeline runs and detect new failures.
    """

    def __init__(
        self,
        gitlab_url: str = "https://gitlab-master.nvidia.com",
        project_path: str = "cudnn/cudnn_frontend",
        private_token: Optional[str] = None,
        verbose: bool = False,
    ):
        self.gitlab_url = gitlab_url.rstrip("/")
        self.project_path = project_path
        self.private_token = private_token or os.environ.get("GITLAB_PRIVATE_TOKEN")
        self.verbose = verbose

        if not self.private_token:
            raise ValueError("GitLab private token required. " "Set GITLAB_PRIVATE_TOKEN environment variable or pass --token")

        self.api_url = f"{self.gitlab_url}/api/v4"
        self.project_id = self._get_project_id()

        self.headers = {
            "PRIVATE-TOKEN": self.private_token,
            "Content-Type": "application/json",
        }

    def _log(self, message: str):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            print(f"[DEBUG] {message}")

    def _get_project_id(self) -> str:
        """Get the project ID from the project path (URL-encoded)"""
        import urllib.parse

        return urllib.parse.quote(self.project_path, safe="")

    def _api_request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make an API request to GitLab"""
        url = f"{self.api_url}{endpoint}"
        self._log(f"API Request: {url}")

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response: {e.response.text}")
            raise

    def get_scheduled_pipelines(self, ref: str = "develop", count: int = 2) -> List[Pipeline]:
        """
        Get the last N scheduled pipeline runs for a branch.

        Args:
            ref: Branch name (default: develop)
            count: Number of pipelines to fetch (default: 2)

        Returns:
            List of Pipeline objects
        """
        self._log(f"Fetching last {count} scheduled pipelines for ref={ref}")

        params = {
            "ref": ref,
            "source": "schedule",
            "per_page": count,
            "order_by": "id",
            "sort": "desc",
        }

        endpoint = f"/projects/{self.project_id}/pipelines"
        data = self._api_request(endpoint, params)

        pipelines = []
        for p in data[:count]:
            pipeline = Pipeline(
                id=p["id"],
                ref=p["ref"],
                status=p["status"],
                source=p["source"],
                created_at=p["created_at"],
                web_url=p["web_url"],
            )
            pipelines.append(pipeline)

        self._log(f"Found {len(pipelines)} pipelines")
        return pipelines

    def get_pipeline_jobs(self, pipeline_id: int) -> List[Job]:
        """
        Get all jobs for a pipeline.

        Args:
            pipeline_id: GitLab pipeline ID

        Returns:
            List of Job objects
        """
        self._log(f"Fetching jobs for pipeline {pipeline_id}")

        jobs = []
        page = 1
        per_page = 100

        while True:
            params = {"per_page": per_page, "page": page}
            endpoint = f"/projects/{self.project_id}/pipelines/{pipeline_id}/jobs"
            data = self._api_request(endpoint, params)

            if not data:
                break

            for j in data:
                job = Job(
                    id=j["id"],
                    name=j["name"],
                    status=j["status"],
                    stage=j["stage"],
                    web_url=j["web_url"],
                    failure_reason=j.get("failure_reason"),
                    allow_failure=j.get("allow_failure", False),
                )
                jobs.append(job)

            if len(data) < per_page:
                break
            page += 1

        self._log(f"Found {len(jobs)} jobs, {len([j for j in jobs if j.is_failed])} failed")
        return jobs

    def compare_pipelines(self, current: Pipeline, previous: Pipeline) -> Tuple[List[Job], List[Job], List[Job]]:
        """
        Compare two pipelines and identify new, fixed, and persistent failures.

        Args:
            current: The more recent pipeline
            previous: The older pipeline

        Returns:
            Tuple of (new_failures, fixed_failures, persistent_failures)
        """
        current_failed = current.failed_job_names
        previous_failed = previous.failed_job_names

        new_failure_names = current_failed - previous_failed
        fixed_failure_names = previous_failed - current_failed
        persistent_failure_names = current_failed & previous_failed

        # Get the actual Job objects
        new_failures = [j for j in current.failed_jobs if j.name in new_failure_names]
        fixed_failures = [j for j in previous.failed_jobs if j.name in fixed_failure_names]
        persistent_failures = [j for j in current.failed_jobs if j.name in persistent_failure_names]

        return new_failures, fixed_failures, persistent_failures

    def analyze_nightly_runs(self, ref: str = "develop") -> Dict:
        """
        Main analysis function - compares the last two nightly runs.

        Args:
            ref: Branch name to analyze

        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*60}")
        print(f"GitLab Pipeline Monitor - cudnn_frontend")
        print(f"{'='*60}")
        print(f"Branch: {ref}")
        print(f"Source: Scheduled (nightly) pipelines")
        print(f"{'='*60}\n")

        # Get last two scheduled pipelines
        pipelines = self.get_scheduled_pipelines(ref=ref, count=2)

        if len(pipelines) < 2:
            print("ERROR: Not enough scheduled pipelines found")
            return {"error": "Not enough pipelines"}

        current_pipeline = pipelines[0]
        previous_pipeline = pipelines[1]

        # Fetch jobs for both pipelines
        print("Fetching pipeline data...")
        current_pipeline.jobs = self.get_pipeline_jobs(current_pipeline.id)
        previous_pipeline.jobs = self.get_pipeline_jobs(previous_pipeline.id)

        # Print pipeline info
        print(f"\n📊 Pipeline Comparison:")
        print(f"  Current:  #{current_pipeline.id} ({current_pipeline.created_at})")
        print(f"            Status: {current_pipeline.status}")
        print(f"            URL: {current_pipeline.web_url}")
        print(f"  Previous: #{previous_pipeline.id} ({previous_pipeline.created_at})")
        print(f"            Status: {previous_pipeline.status}")
        print(f"            URL: {previous_pipeline.web_url}")

        # Compare pipelines
        new_failures, fixed_failures, persistent_failures = self.compare_pipelines(current_pipeline, previous_pipeline)

        results = {
            "current_pipeline": {
                "id": current_pipeline.id,
                "status": current_pipeline.status,
                "created_at": current_pipeline.created_at,
                "web_url": current_pipeline.web_url,
                "total_jobs": len(current_pipeline.jobs),
                "failed_jobs": len(current_pipeline.failed_jobs),
            },
            "previous_pipeline": {
                "id": previous_pipeline.id,
                "status": previous_pipeline.status,
                "created_at": previous_pipeline.created_at,
                "web_url": previous_pipeline.web_url,
                "total_jobs": len(previous_pipeline.jobs),
                "failed_jobs": len(previous_pipeline.failed_jobs),
            },
            "new_failures": [{"name": j.name, "stage": j.stage, "url": j.web_url} for j in new_failures],
            "fixed_failures": [{"name": j.name, "stage": j.stage} for j in fixed_failures],
            "persistent_failures": [{"name": j.name, "stage": j.stage, "url": j.web_url} for j in persistent_failures],
        }

        # Print results
        print(f"\n{'='*60}")
        print("ANALYSIS RESULTS")
        print(f"{'='*60}")

        # New failures (most important!)
        print(f"\n🚨 NEW FAILURES ({len(new_failures)}):")
        if new_failures:
            for job in sorted(new_failures, key=lambda j: (j.stage, j.name)):
                print(f"   ❌ [{job.stage}] {job.name}")
                print(f"      URL: {job.web_url}")
                if job.failure_reason:
                    print(f"      Reason: {job.failure_reason}")
        else:
            print("   ✅ No new failures!")

        # Fixed failures
        print(f"\n✅ FIXED ({len(fixed_failures)}):")
        if fixed_failures:
            for job in sorted(fixed_failures, key=lambda j: (j.stage, j.name)):
                print(f"   🔧 [{job.stage}] {job.name}")
        else:
            print("   (none)")

        # Persistent failures
        print(f"\n⚠️  PERSISTENT FAILURES ({len(persistent_failures)}):")
        if persistent_failures:
            for job in sorted(persistent_failures, key=lambda j: (j.stage, j.name)):
                print(f"   🔴 [{job.stage}] {job.name}")
                print(f"      URL: {job.web_url}")
        else:
            print("   ✅ No persistent failures!")

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  Current pipeline:  {len(current_pipeline.failed_jobs)} failed / {len(current_pipeline.jobs)} total")
        print(f"  Previous pipeline: {len(previous_pipeline.failed_jobs)} failed / {len(previous_pipeline.jobs)} total")
        print(f"  New failures:      {len(new_failures)}")
        print(f"  Fixed:             {len(fixed_failures)}")
        print(f"  Persistent:        {len(persistent_failures)}")

        if new_failures:
            print(f"\n⚠️  ACTION REQUIRED: {len(new_failures)} new failure(s) detected!")
        else:
            print(f"\n✅ All clear: No new failures in the latest nightly run.")

        print(f"{'='*60}\n")

        return results

    def get_failure_log(self, job_id: int) -> str:
        """
        Get the log/trace for a failed job.

        Args:
            job_id: GitLab job ID

        Returns:
            Job trace/log as string
        """
        endpoint = f"/projects/{self.project_id}/jobs/{job_id}/trace"
        url = f"{self.api_url}{endpoint}"

        response = requests.get(url, headers=self.headers, timeout=60)
        response.raise_for_status()

        return response.text


def main():
    parser = argparse.ArgumentParser(description="Monitor GitLab nightly pipeline runs and detect new failures")
    parser.add_argument("--token", help="GitLab private token (or set GITLAB_PRIVATE_TOKEN env var)")
    parser.add_argument(
        "--gitlab-url",
        default="https://gitlab-master.nvidia.com",
        help="GitLab instance URL",
    )
    parser.add_argument("--project", default="cudnn/cudnn_frontend", help="GitLab project path")
    parser.add_argument("--ref", default="develop", help="Branch to monitor (default: develop)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    try:
        monitor = GitLabPipelineMonitor(
            gitlab_url=args.gitlab_url,
            project_path=args.project,
            private_token=args.token,
            verbose=args.verbose,
        )

        results = monitor.analyze_nightly_runs(ref=args.ref)

        if args.json:
            print(json.dumps(results, indent=2))

        # Exit with non-zero if there are new failures
        if results.get("new_failures"):
            sys.exit(1)

    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
