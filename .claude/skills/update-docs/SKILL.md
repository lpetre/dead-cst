---
name: update-docs
description: Use when the user wants to make sure documentation is up-to-date with the changes on the current branch — README, CHANGELOG, docstrings, and PR title/description. Trigger on phrasings like "make sure the docs are updated", "update the docs", "check docs", or after a feature/fix is complete and before merging.
---

# update-docs

Audit and update the four documentation surfaces that need to stay in sync with code changes on the current branch:

1. **README.md** — user-facing feature lists, install/usage examples, public API mentions.
2. **CHANGELOG.md** — a `[Unreleased]` entry for any user-visible change (new feature, bug fix, deprecation, removal, behavior change). Follow the existing Keep-a-Changelog sections (`Added` / `Changed` / `Fixed` / `Removed` / `Deprecated`). Skip purely internal refactors that no consumer can observe.
3. **Docstrings** — every symbol added or whose behavior was changed in this branch. Prefer updating an existing docstring over writing a new module-level comment. Match the surrounding style (this codebase favors short, no-nonsense docstrings; do not invent multi-paragraph essays).
4. **PR title and description** — if a PR exists, make sure the title summarizes the change (under ~70 chars) and the body covers Summary + Test plan and reflects everything actually in the diff (not just the first commit).

## How to run the skill

1. **Find what changed.** Determine the branch's base (usually `main`) and run `git diff <base>...HEAD --stat` plus `git log <base>..HEAD --oneline` to see the full set of changes — not just the latest commit.
2. **For each surface, check then act:**
   - Read README.md and decide whether any of the changed/added behavior belongs there. Edit if so.
   - Read CHANGELOG.md's `[Unreleased]` block. If a user-visible change is missing an entry, add one in the right subsection. If the section doesn't exist yet, create it. Match the wrapping/voice of the existing entries.
   - For each new or modified public symbol (anything reachable from `dead_cst/__init__.py` or a non-`_`-prefixed submodule), open the file and confirm the docstring is accurate. Update or add as needed. Skip private (`_`-prefixed) modules unless the user asks.
   - If a PR exists for this branch (`gh pr view` or the GitHub MCP tools), compare the title/body against the diff and update via `mcp__github__update_pull_request` if they're stale or missing the new changes. Only do this if a PR is already open — do **not** create a PR.
3. **Report back** with a short bulleted summary: which files you touched, which surfaces were already up to date, and any items you intentionally skipped (with the reason).

## Guardrails

- Never amend prior commits — make a new commit with the doc updates.
- Don't add CHANGELOG entries for changes you can't see in the diff.
- Don't create a PR if none exists. Don't push unless the user asks.
- Respect the project's "no comments unless they explain non-obvious why" rule from CLAUDE.md — that applies to docstrings too. Short and accurate beats long and aspirational.
