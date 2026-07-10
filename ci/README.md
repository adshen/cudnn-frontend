# GitLab CI layout

The root `.gitlab-ci.yml` defines pipeline-wide behavior, includes the CI modules below, and declares stage order. Keep job implementations out of the root file.

## Where things live

- `common/`: reusable templates shared by stages.
  - `rules.yml`: common job rules.
  - `images.yml`: container image templates.
  - `matrices.yml`: runner and architecture matrices.
  - `cudnn.yml`: cuDNN artifact setup.
- `stages/<name>/`: jobs and scripts owned by one test or delivery area. `jobs.yml` contains that area's GitLab configuration; supporting scripts sit beside it.
- `qa_matrix/`: scheduled compatibility-matrix jobs parameterized by CUDA and cuDNN versions.
- `manual/`: scripts not referenced by the active GitLab configuration. See `manual/README.md` before using or removing them.

## Add or change a job

1. Find the matching directory under `stages/`.
2. Update its `jobs.yml`; keep stage-specific scripts in the same directory.
3. Put only genuinely cross-stage templates in `common/`.
4. For a new stage area, add its `jobs.yml` to the root include list and add the stage name to the root stage list if needed.
5. Run GitLab CI lint, `git diff --check`, and syntax checks for changed scripts.

Paths in job commands are resolved from the repository root. Moving a script therefore requires updating its callers in `jobs.yml` and any references inside the script.
