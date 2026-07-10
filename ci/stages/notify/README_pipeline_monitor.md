# GitLab Pipeline Monitor Agent

Monitor cudnn_frontend nightly pipeline runs and detect new failures between runs.

## Setup

### 1. Get GitLab Token

1. Go to [GitLab Personal Access Tokens](https://gitlab-master.nvidia.com/-/user_settings/personal_access_tokens)
2. Create a new token with `read_api` scope
3. Set the environment variable:

```bash
export GITLAB_PRIVATE_TOKEN=your_token_here
```

### 2. Install Dependencies

```bash
pip install requests
```

## Usage

### Command Line

```bash
# Basic usage
python gitlab_pipeline_monitor.py

# With verbose output
python gitlab_pipeline_monitor.py --verbose

# Output as JSON
python gitlab_pipeline_monitor.py --json

# Check a different branch
python gitlab_pipeline_monitor.py --ref main
```

### As a Python Module

```python
from gitlab_pipeline_monitor import GitLabPipelineMonitor

monitor = GitLabPipelineMonitor(
    gitlab_url="https://gitlab-master.nvidia.com",
    project_path="cudnn/cudnn_frontend",
    private_token="your_token_here",
    verbose=False,
)

result = monitor.analyze_nightly_runs(ref="develop")
print(result)
```

## Output Example

```
============================================================
GitLab Pipeline Monitor - cudnn_frontend
============================================================
Branch: develop
Source: Scheduled (nightly) pipelines
============================================================

Fetching pipeline data...

📊 Pipeline Comparison:
  Current:  #12345 (2025-03-17T02:00:00Z)
            Status: failed
            URL: https://gitlab-master.nvidia.com/cudnn/cudnn_frontend/-/pipelines/12345
  Previous: #12340 (2025-03-16T02:00:00Z)
            Status: success
            URL: https://gitlab-master.nvidia.com/cudnn/cudnn_frontend/-/pipelines/12340

============================================================
ANALYSIS RESULTS
============================================================

🚨 NEW FAILURES (2):
   ❌ [test] L4_py_samples
      URL: https://gitlab-master.nvidia.com/cudnn/cudnn_frontend/-/jobs/67890
   ❌ [test] L4_cpp_samples
      URL: https://gitlab-master.nvidia.com/cudnn/cudnn_frontend/-/jobs/67891

✅ FIXED (1):
   🔧 [build] build_windows

⚠️  PERSISTENT FAILURES (0):
   ✅ No persistent failures!

============================================================
SUMMARY
============================================================
  Current pipeline:  2 failed / 45 total
  Previous pipeline: 1 failed / 45 total
  New failures:      2
  Fixed:             1
  Persistent:        0

⚠️  ACTION REQUIRED: 2 new failure(s) detected!
============================================================
```

## Exit Codes

- `0`: Success, no new failures
- `1`: New failures detected
- `2`: Error (configuration, API, etc.)

## Files

- `gitlab_pipeline_monitor.py`: Main monitoring agent with full functionality
- `pipeline_slack_notifier.py`: Slack notification script
- `README_pipeline_monitor.md`: This documentation

---

## Slack Integration

### Step 1: Create Slack Webhook

1. Go to [Slack API Apps](https://api.slack.com/apps)
2. Create a new app (or use existing)
3. Enable "Incoming Webhooks"
4. Add a webhook to your workspace
5. Select the channel (e.g., `#cudnn-frontend-ci`)
6. Copy the webhook URL

### Step 2: Add GitLab CI/CD Variables

Go to **Settings > CI/CD > Variables** in your GitLab project:

| Variable | Value | Options |
|----------|-------|---------|
| `SLACK_WEBHOOK_URL` | `https://hooks.slack.com/services/...` | Masked, Protected |
| `GITLAB_PRIVATE_TOKEN` | Your GitLab token | Masked, Protected |

### Step 3: Add Job to .gitlab-ci.yml

Add a `notify` stage to your pipeline:

```yaml
stages:
  - build
  - test
  - python_samples
  - notify  # Add this

slack_notification:
  stage: notify
  image: python:3.10-slim
  variables:
    GIT_STRATEGY: none
  before_script:
    - pip install requests --quiet
  script:
    - |
      python3 << 'EOF'
      import os, requests, json
      
      GITLAB_URL = os.environ["CI_SERVER_URL"]
      PROJECT_ID = os.environ["CI_PROJECT_ID"]
      PIPELINE_ID = os.environ["CI_PIPELINE_ID"]
      PIPELINE_URL = os.environ["CI_PIPELINE_URL"]
      REF = os.environ["CI_COMMIT_REF_NAME"]
      TOKEN = os.environ["GITLAB_PRIVATE_TOKEN"]
      SLACK_WEBHOOK = os.environ["SLACK_WEBHOOK_URL"]
      
      headers = {"PRIVATE-TOKEN": TOKEN}
      
      # Get jobs
      jobs = requests.get(
          f"{GITLAB_URL}/api/v4/projects/{PROJECT_ID}/pipelines/{PIPELINE_ID}/jobs",
          headers=headers, params={"per_page": 100}
      ).json()
      
      failed = [j for j in jobs if j["status"] == "failed" and not j.get("allow_failure")]
      
      # Get previous pipeline for comparison
      pipelines = requests.get(
          f"{GITLAB_URL}/api/v4/projects/{PROJECT_ID}/pipelines",
          headers=headers, params={"ref": REF, "source": "schedule", "per_page": 2}
      ).json()
      
      prev_failed = set()
      if len(pipelines) > 1:
          prev_jobs = requests.get(
              f"{GITLAB_URL}/api/v4/projects/{PROJECT_ID}/pipelines/{pipelines[1]['id']}/jobs",
              headers=headers, params={"per_page": 100}
          ).json()
          prev_failed = {j["name"] for j in prev_jobs if j["status"] == "failed"}
      
      new_failures = {j["name"] for j in failed} - prev_failed
      
      # Send to Slack
      color = "danger" if new_failures else ("warning" if failed else "good")
      status = f"🚨 {len(new_failures)} NEW" if new_failures else ("⚠️ Persistent" if failed else "✅ Passing")
      
      text = f"*cudnn_frontend* `{REF}` <{PIPELINE_URL}|#{PIPELINE_ID}>\n*{status}*"
      if new_failures:
          text += "\n" + "\n".join(f"• `{n}`" for n in new_failures)
      
      requests.post(SLACK_WEBHOOK, json={"attachments": [{"color": color, "text": text}]})
      EOF
  rules:
    - if: '$CI_PIPELINE_SOURCE == "schedule"'
      when: always
  when: always
  allow_failure: true
```

### Manual Testing

Test the Slack notifier manually:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
export GITLAB_PRIVATE_TOKEN="glpat-xxx"

# Dry run (prints instead of sending)
python pipeline_slack_notifier.py --dry-run

# Send to Slack
python pipeline_slack_notifier.py

# Only notify if there are NEW failures
python pipeline_slack_notifier.py --only-failures
```

### Slack Message Example

The notification will look like:

```
🚨 cudnn_frontend Pipeline Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Branch: develop
Pipeline: #39953675
Status: failed
Jobs: 4 failed / 84 total

🔴 New Failures (2):
• python_samples:cudnn_ga_latest: [Blackwell] - View Job
• python_samples:cudnn_rel_latest: [Hopper] - View Job

✅ Fixed (1):
• build_windows
```

## API Reference

### GitLabPipelineMonitor Class

```python
monitor = GitLabPipelineMonitor(
    gitlab_url="https://gitlab-master.nvidia.com",
    project_path="cudnn/cudnn_frontend",
    private_token="your_token",
    verbose=False
)

# Get scheduled pipelines
pipelines = monitor.get_scheduled_pipelines(ref="develop", count=2)

# Get jobs for a pipeline
jobs = monitor.get_pipeline_jobs(pipeline_id=12345)

# Compare two pipelines
new, fixed, persistent = monitor.compare_pipelines(current, previous)

# Full analysis
results = monitor.analyze_nightly_runs(ref="develop")

# Get failure log
log = monitor.get_failure_log(job_id=67890)
```

### pipeline_agent Functions

```python
# Full status check
result = check_pipeline_status(token=None, ref="develop", verbose=False)

# Get new failures only
failures = get_new_failures(token=None, ref="develop")

# Get summary string
summary = get_pipeline_summary(token=None, ref="develop")

# Format failures as report
report = format_failure_report(failures, title="Failures")
```
