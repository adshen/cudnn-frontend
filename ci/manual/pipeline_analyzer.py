#!/usr/bin/env python3


## Usage
## python gitlab_pipeline_compare.py --token glpat-<...> --pipeline1 28672341 --pipeline2 28633133


import requests
import argparse
import json
from collections import defaultdict

cudnn_project_id = 28254


def get_test_report(gitlab_ci_token, pipeline_id):
    """Fetch test report for a GitLab pipeline"""

    url = f"https://gitlab-master.nvidia.com/api/v4/projects/{cudnn_project_id}/pipelines/{pipeline_id}/test_report"
    headers = {"PRIVATE-TOKEN": gitlab_ci_token}

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error fetching pipeline {pipeline_id}: {response.status_code}")
        print(f"Response: {response.text}")
        return None

    return response.json()


def extract_failed_tests(test_report):
    """Extract failed tests from a test report"""

    if not test_report or "test_suites" not in test_report:
        return {}

    failed_tests = {}

    for suite in test_report["test_suites"]:
        suite_name = suite["name"]
        for test_case in suite.get("test_cases", []):
            if test_case["status"] == "failed":
                test_name = test_case["name"]
                key = f"{suite_name}::{test_name}"
                failed_tests[key] = {
                    "suite": suite_name,
                    "name": test_name,
                    "failure": test_case.get("failure", {}).get("message", "No message"),
                    "execution_time": test_case.get("execution_time", 0),
                }

    return failed_tests


def compare_failed_tests(gitlab_ci_token, pipeline1_id, pipeline2_id):
    """Compare failed tests between two pipelines"""

    print(f"Fetching test reports...")

    # Get test reports
    report1 = get_test_report(gitlab_ci_token, pipeline1_id)
    report2 = get_test_report(gitlab_ci_token, pipeline2_id)

    if not report1 or not report2:
        print("Failed to fetch test reports. Check pipeline IDs and token.")
        return

    # Get failed tests
    failed1 = extract_failed_tests(report1)
    failed2 = extract_failed_tests(report2)

    print(f"\nPipeline {pipeline1_id}: {len(failed1)} failed tests")
    print(f"Pipeline {pipeline2_id}: {len(failed2)} failed tests")

    # Find differences
    unique_to_pipeline1 = set(failed1.keys()) - set(failed2.keys())
    unique_to_pipeline2 = set(failed2.keys()) - set(failed1.keys())
    common_failures = set(failed1.keys()) & set(failed2.keys())

    # Prepare results
    results = {
        "summary": {
            "pipeline1_id": pipeline1_id,
            "pipeline2_id": pipeline2_id,
            "failed_in_pipeline1": len(failed1),
            "failed_in_pipeline2": len(failed2),
            "unique_to_pipeline1": len(unique_to_pipeline1),
            "unique_to_pipeline2": len(unique_to_pipeline2),
            "common_failures": len(common_failures),
        },
        "unique_to_pipeline1": [failed1[test] for test in unique_to_pipeline1],
        "unique_to_pipeline2": [failed2[test] for test in unique_to_pipeline2],
        "common_failures": [failed1[test] for test in common_failures],
    }

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Tests failing only in pipeline {pipeline1_id}: {len(unique_to_pipeline1)}")
    print(f"Tests failing only in pipeline {pipeline2_id}: {len(unique_to_pipeline2)}")
    print(f"Tests failing in both pipelines: {len(common_failures)}")

    # Print details of unique failures
    if unique_to_pipeline1:
        print(f"\n=== TESTS FAILING ONLY IN PIPELINE {pipeline1_id} ===")
        for i, test_key in enumerate(sorted(unique_to_pipeline1), 1):
            test = failed1[test_key]
            print(f"{i}. {test['suite']}::{test['name']}")

    if unique_to_pipeline2:
        print(f"\n=== TESTS FAILING ONLY IN PIPELINE {pipeline2_id} ===")
        for i, test_key in enumerate(sorted(unique_to_pipeline2), 1):
            test = failed2[test_key]
            print(f"{i}. {test['suite']}::{test['name']}")

    # Save results to file
    output_file = f"pipeline_comparison_{pipeline1_id}_vs_{pipeline2_id}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare failed tests between two GitLab pipelines")

    parser.add_argument("--pipeline1", type=int, help="ID of the first pipeline")
    parser.add_argument("--pipeline2", type=int, help="ID of the second pipeline")

    parser.add_argument("--token", required=True, help="GitLab API token (e.g., glpat-xxx)")

    args = parser.parse_args()

    gitlab_ci_token = args.token

    compare_failed_tests(gitlab_ci_token, args.pipeline1, args.pipeline2)
