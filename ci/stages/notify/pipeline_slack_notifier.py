#!/usr/bin/env python3
"""
Pipeline Slack Notifier

Posts pipeline status to Slack when run as part of GitLab CI.
Can compare current pipeline with previous run to identify new failures.

Usage:
    # As part of GitLab CI (uses CI environment variables)
    python pipeline_slack_notifier.py

    # Manual run with specific pipeline
    python pipeline_slack_notifier.py --pipeline-id 39953675

Environment Variables Required:
    SLACK_WEBHOOK_URL: Slack incoming webhook URL
    GITLAB_PRIVATE_TOKEN: GitLab API token (or CI_JOB_TOKEN in CI)

GitLab CI Variables (automatically set):
    CI_PIPELINE_ID: Current pipeline ID
    CI_PIPELINE_URL: Pipeline URL
    CI_PROJECT_PATH: Project path
    CI_COMMIT_REF_NAME: Branch name
    CI_PIPELINE_SOURCE: Pipeline trigger source
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)

# Add script directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from gitlab_pipeline_monitor import GitLabPipelineMonitor, Pipeline


class SlackNotifier:
    """Send notifications to Slack via webhook or Bot API."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        bot_token: Optional[str] = None,
        channel: Optional[str] = None,
    ):
        self.webhook_url = webhook_url
        self.bot_token = bot_token  # xoxb-... token
        self.channel = channel  # e.g., "#cudnn-frontend-ci" or "C01234567"

        if not webhook_url and not bot_token:
            raise ValueError("Either webhook_url or bot_token must be provided")

        if bot_token and not channel:
            raise ValueError("channel is required when using bot_token")

    def send_message(self, message: Dict) -> bool:
        """Send a message to Slack."""
        if self.bot_token:
            return self._send_via_bot_api(message)
        else:
            return self._send_via_webhook(message)

    def _send_via_webhook(self, message: Dict) -> bool:
        """Send message via Incoming Webhook."""
        # Add channel override if specified
        if self.channel:
            message["channel"] = self.channel

        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to send Slack message via webhook: {e}")
            return False

    def _send_via_bot_api(self, message: Dict) -> bool:
        """Send message via Slack Bot API (chat.postMessage)."""
        # Convert webhook-style message to Bot API format
        payload = {
            "channel": self.channel,
        }

        # Handle attachments format
        if "attachments" in message:
            payload["attachments"] = message["attachments"]
            # Extract text from attachment if present
            if message["attachments"] and "text" in message["attachments"][0]:
                payload["text"] = "Pipeline Status Update"  # Fallback text for notifications
        elif "text" in message:
            payload["text"] = message["text"]

        try:
            response = requests.post(
                "https://slack.com/api/chat.postMessage",
                headers={
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )

            result = response.json()
            if not result.get("ok"):
                print(f"Slack API error: {result.get('error')}")
                if result.get("error") == "channel_not_found":
                    print(f"  → Make sure the bot is added to channel: {self.channel}")
                elif result.get("error") == "not_in_channel":
                    print(f"  → Invite the bot to the channel first: /invite @YourBotName")
                return False

            print(f"✅ Posted to Slack channel: {self.channel}")
            return True

        except Exception as e:
            print(f"Failed to send Slack message via Bot API: {e}")
            return False

    def send_pipeline_report(
        self,
        current_pipeline: Dict,
        previous_pipeline: Optional[Dict],
        new_failures: List[Dict],
        fixed_failures: List[Dict],
        persistent_failures: List[Dict],
        branch: str = "develop",
    ) -> bool:
        """Send a formatted pipeline report to Slack matching terminal output style."""

        # Determine overall status and color
        if new_failures:
            color = "danger"  # Red
        elif persistent_failures:
            color = "warning"  # Yellow
        else:
            color = "good"  # Green

        # Build the message in the same format as terminal output
        lines = []

        # Header
        lines.append("=" * 50)
        lines.append("*GitLab Pipeline Monitor - cudnn_frontend*")
        lines.append("=" * 50)
        lines.append(f"*Branch:* `{branch}`")
        lines.append(f"*Source:* Scheduled (nightly) pipelines")
        lines.append("=" * 50)
        lines.append("")

        # Pipeline Comparison
        lines.append("📊 *Pipeline Comparison:*")
        lines.append(f"  *Current:*  <{current_pipeline['web_url']}|#{current_pipeline['id']}>")
        lines.append(f"            Status: `{current_pipeline['status']}`")

        if previous_pipeline:
            lines.append(f"  *Previous:* <{previous_pipeline['web_url']}|#{previous_pipeline['id']}>")

        lines.append("")
        lines.append("=" * 50)
        lines.append("*ANALYSIS RESULTS*")
        lines.append("=" * 50)
        lines.append("")

        # New Failures
        lines.append(f"🚨 *NEW FAILURES ({len(new_failures)}):*")
        if new_failures:
            for f in new_failures:
                lines.append(f"   ❌ [{f.get('stage', 'unknown')}] `{f['name']}`")
                if f.get("url"):
                    lines.append(f"      <{f['url']}|View Job>")
        else:
            lines.append("   ✅ No new failures!")
        lines.append("")

        # Fixed
        lines.append(f"✅ *FIXED ({len(fixed_failures)}):*")
        if fixed_failures:
            for f in fixed_failures:
                lines.append(f"   🔧 [{f.get('stage', 'unknown')}] `{f['name']}`")
        else:
            lines.append("   (none)")
        lines.append("")

        # Persistent Failures
        lines.append(f"⚠️  *PERSISTENT FAILURES ({len(persistent_failures)}):*")
        if persistent_failures:
            for f in persistent_failures:
                lines.append(f"   🔴 [{f.get('stage', 'unknown')}] `{f['name']}`")
                if f.get("url"):
                    lines.append(f"      <{f['url']}|View Job>")
        else:
            lines.append("   ✅ No persistent failures!")
        lines.append("")

        # Summary
        lines.append("=" * 50)
        lines.append("*SUMMARY*")
        lines.append("=" * 50)
        lines.append(f"  Current pipeline:  {current_pipeline.get('failed_jobs', 0)} failed / {current_pipeline.get('total_jobs', 0)} total")
        if previous_pipeline:
            lines.append(f"  Previous pipeline: {previous_pipeline.get('failed_jobs', '?')} failed")
        lines.append(f"  New failures:      {len(new_failures)}")
        lines.append(f"  Fixed:             {len(fixed_failures)}")
        lines.append(f"  Persistent:        {len(persistent_failures)}")
        lines.append("")

        # Final status message
        if new_failures:
            lines.append(f"⚠️  *ACTION REQUIRED:* {len(new_failures)} new failure(s) detected!")
        else:
            lines.append("✅ *All clear:* No new failures in the latest nightly run.")

        lines.append("=" * 50)

        # Join all lines
        text = "\n".join(lines)

        # Build the final message
        message = {"attachments": [{"color": color, "text": text, "mrkdwn_in": ["text"]}]}

        return self.send_message(message)


def get_ci_environment() -> Dict[str, str]:
    """Get GitLab CI environment variables."""
    return {
        "pipeline_id": os.environ.get("CI_PIPELINE_ID"),
        "pipeline_url": os.environ.get("CI_PIPELINE_URL"),
        "project_path": os.environ.get("CI_PROJECT_PATH", "cudnn/cudnn_frontend"),
        "ref": os.environ.get("CI_COMMIT_REF_NAME", "develop"),
        "source": os.environ.get("CI_PIPELINE_SOURCE"),
        "job_token": os.environ.get("CI_JOB_TOKEN"),
        "gitlab_url": os.environ.get("CI_SERVER_URL", "https://gitlab-master.nvidia.com"),
    }


def main():
    parser = argparse.ArgumentParser(description="Post pipeline status to Slack")
    parser.add_argument("--webhook-url", help="Slack webhook URL (or set SLACK_WEBHOOK_URL env var)")
    parser.add_argument(
        "--bot-token",
        help="Slack Bot User OAuth Token (xoxb-...) (or set SLACK_BOT_TOKEN env var)",
    )
    parser.add_argument("--token", help="GitLab private token (or set GITLAB_PRIVATE_TOKEN env var)")
    parser.add_argument(
        "--pipeline-id",
        type=int,
        help="Pipeline ID to report on (default: current CI pipeline)",
    )
    parser.add_argument("--ref", default="develop", help="Branch name (default: develop)")
    parser.add_argument(
        "--channel",
        help="Slack channel to post to (e.g., #cudnn-frontend-ci). Overrides webhook default.",
    )
    parser.add_argument(
        "--only-failures",
        action="store_true",
        help="Only post to Slack if there are new failures",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print message instead of sending to Slack",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Get configuration
    ci_env = get_ci_environment()

    slack_webhook = args.webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    slack_bot_token = args.bot_token or os.environ.get("SLACK_BOT_TOKEN")
    slack_channel = args.channel or os.environ.get("SLACK_CHANNEL")
    gitlab_token = args.token or os.environ.get("GITLAB_PRIVATE_TOKEN") or ci_env["job_token"]
    ref = args.ref or ci_env["ref"]

    if not slack_webhook and not slack_bot_token and not args.dry_run:
        print("Error: Slack webhook URL or Bot Token required")
        print("Options:")
        print("  1. Set SLACK_WEBHOOK_URL environment variable or use --webhook-url")
        print("  2. Set SLACK_BOT_TOKEN environment variable or use --bot-token (requires --channel)")
        sys.exit(1)

    if slack_bot_token and not slack_channel and not args.dry_run:
        print("Error: --channel is required when using --bot-token")
        print("Example: --channel '#cudnn-frontend-ci'")
        sys.exit(1)

    if not gitlab_token:
        print("Error: GitLab token required")
        print("Set GITLAB_PRIVATE_TOKEN environment variable or use --token")
        sys.exit(1)

    try:
        # Initialize monitor
        monitor = GitLabPipelineMonitor(
            gitlab_url=ci_env["gitlab_url"],
            project_path=ci_env["project_path"],
            private_token=gitlab_token,
            verbose=args.verbose,
        )

        # Get pipelines to compare
        print(f"Fetching pipeline data for {ref}...")
        pipelines = monitor.get_scheduled_pipelines(ref=ref, count=2)

        if not pipelines:
            print("No scheduled pipelines found")
            sys.exit(1)

        current = pipelines[0]
        previous = pipelines[1] if len(pipelines) > 1 else None

        # Get jobs
        current.jobs = monitor.get_pipeline_jobs(current.id)
        if previous:
            previous.jobs = monitor.get_pipeline_jobs(previous.id)

        # Compare
        if previous:
            new_failures, fixed_failures, persistent_failures = monitor.compare_pipelines(current, previous)
        else:
            new_failures = current.failed_jobs
            fixed_failures = []
            persistent_failures = []

        # Convert to dicts for Slack
        new_failures_dict = [{"name": j.name, "stage": j.stage, "url": j.web_url} for j in new_failures]
        fixed_failures_dict = [{"name": j.name, "stage": j.stage} for j in fixed_failures]
        persistent_failures_dict = [{"name": j.name, "stage": j.stage, "url": j.web_url} for j in persistent_failures]

        current_dict = {
            "id": current.id,
            "status": current.status,
            "web_url": current.web_url,
            "failed_jobs": len(current.failed_jobs),
            "total_jobs": len(current.jobs),
        }

        previous_dict = {"id": previous.id, "status": previous.status, "web_url": previous.web_url} if previous else None

        # Print summary
        print(f"\n{'='*50}")
        print(f"Pipeline #{current.id}: {current.status}")
        print(f"New failures: {len(new_failures_dict)}")
        print(f"Fixed: {len(fixed_failures_dict)}")
        print(f"Persistent: {len(persistent_failures_dict)}")
        print(f"{'='*50}\n")

        # Check if we should post
        if args.only_failures and not new_failures_dict:
            print("No new failures - skipping Slack notification")
            sys.exit(0)

        # Send to Slack
        if args.dry_run:
            print("\nDRY RUN - Would send to Slack:")
            print("=" * 60)

            # Build the same message format to preview
            notifier = SlackNotifier("dry-run")

            # Generate preview text
            lines = []
            lines.append("=" * 50)
            lines.append("GitLab Pipeline Monitor - cudnn_frontend")
            lines.append("=" * 50)
            lines.append(f"Branch: {ref}")
            lines.append("Source: Scheduled (nightly) pipelines")
            lines.append("=" * 50)
            lines.append("")
            lines.append("📊 Pipeline Comparison:")
            lines.append(f"  Current:  #{current_dict['id']}")
            lines.append(f"            Status: {current_dict['status']}")
            lines.append(f"            URL: {current_dict['web_url']}")
            if previous_dict:
                lines.append(f"  Previous: #{previous_dict['id']}")
                lines.append(f"            URL: {previous_dict['web_url']}")
            lines.append("")
            lines.append("=" * 50)
            lines.append("ANALYSIS RESULTS")
            lines.append("=" * 50)
            lines.append("")
            lines.append(f"🚨 NEW FAILURES ({len(new_failures_dict)}):")
            if new_failures_dict:
                for f in new_failures_dict:
                    lines.append(f"   ❌ [{f.get('stage', 'unknown')}] {f['name']}")
                    if f.get("url"):
                        lines.append(f"      URL: {f['url']}")
            else:
                lines.append("   ✅ No new failures!")
            lines.append("")
            lines.append(f"✅ FIXED ({len(fixed_failures_dict)}):")
            if fixed_failures_dict:
                for f in fixed_failures_dict:
                    lines.append(f"   🔧 [{f.get('stage', 'unknown')}] {f['name']}")
            else:
                lines.append("   (none)")
            lines.append("")
            lines.append(f"⚠️  PERSISTENT FAILURES ({len(persistent_failures_dict)}):")
            if persistent_failures_dict:
                for f in persistent_failures_dict:
                    lines.append(f"   🔴 [{f.get('stage', 'unknown')}] {f['name']}")
                    if f.get("url"):
                        lines.append(f"      URL: {f['url']}")
            else:
                lines.append("   ✅ No persistent failures!")
            lines.append("")
            lines.append("=" * 50)
            lines.append("SUMMARY")
            lines.append("=" * 50)
            lines.append(f"  Current pipeline:  {current_dict.get('failed_jobs', 0)} failed / {current_dict.get('total_jobs', 0)} total")
            if previous_dict:
                lines.append(f"  Previous pipeline: (compared)")
            lines.append(f"  New failures:      {len(new_failures_dict)}")
            lines.append(f"  Fixed:             {len(fixed_failures_dict)}")
            lines.append(f"  Persistent:        {len(persistent_failures_dict)}")
            lines.append("")
            if new_failures_dict:
                lines.append(f"⚠️  ACTION REQUIRED: {len(new_failures_dict)} new failure(s) detected!")
            else:
                lines.append("✅ All clear: No new failures in the latest nightly run.")
            lines.append("=" * 50)

            print("\n".join(lines))
            print("=" * 60)
        else:
            notifier = SlackNotifier(
                webhook_url=slack_webhook,
                bot_token=slack_bot_token,
                channel=slack_channel,
            )
            success = notifier.send_pipeline_report(
                current_pipeline=current_dict,
                previous_pipeline=previous_dict,
                new_failures=new_failures_dict,
                fixed_failures=fixed_failures_dict,
                persistent_failures=persistent_failures_dict,
                branch=ref,
            )

            if success:
                print("✅ Slack notification sent successfully!")
            else:
                print("❌ Failed to send Slack notification")
                sys.exit(1)

        # Exit with error code if new failures
        if new_failures_dict:
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
