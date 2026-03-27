---
name: repo-release-curation
description: Curate and restructure a near-publication ML repository for clean release, reproducible runs, and minimal layout while preserving eval/data and paper assets.
---

# Repo Release Curation

Use this skill when asked to tidy, restructure, or package a near-publication repository for open-source release while preserving reproducibility.

## Workflow

1) **Confirm preservation scope**
   - Always keep: `acl26_instructDiff/` (paper source), `eval/`, `data/`.
   - Confirm any other keep/delete constraints before removing files.

2) **Minimize root layout**
   - Target root: `acl26_instructDiff/`, `configs/`, `data/`, `eval/`, `src/`, `README.md`, `requirements.txt`, `setup.py`, `pyproject.toml`, `.gitignore`.
   - Move all code into `src/` and adjust imports accordingly.

3) **Consolidate tools**
   - Merge plotting/analysis into `src/<tool_pkg>/analysis/`.
   - If utility helpers exist, merge into package `__init__.py` or a single module to reduce file count.

4) **Packaging**
   - Add `requirements.txt`, `setup.py`, and `pyproject.toml`.
   - Provide `console_scripts` entry for CLI (`instdiff`).
   - Ensure `pip install -e .` works without extra steps.

5) **Docs (bilingual)**
   - Write a single `README.md` with Chinese + English sections.
   - Include pipeline overview, quick start, config usage, and **actual experiment results**.
   - Reference paper figures using local assets.

6) **Configs and outputs**
   - Keep configs under `configs/<domain>/`.
   - Default outputs go to `runs/` and are ignored via `.gitignore`.

7) **AI trace removal**
   - Remove assistant-specific logs or meta files.
   - Remove temporary logs or internal notes not needed for users.

## Checks

- `pip install -e .` installs `instdiff` CLI.
- `README.md` is the only root doc, bilingual, and includes results.
- Root contains only required top-level folders.
- `eval/` and `data/` are preserved untouched.
