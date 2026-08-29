#!/usr/bin/env bash
# ============================================================================
# deploy/config.sh — the ONLY file that differs between projects.
#
# Everything else in deploy/ is written to be copied verbatim to another repo.
# When porting these scripts, copy the whole deploy/ dir and rewrite this file.
# ============================================================================

# --- identity ---------------------------------------------------------------
PROJECT_NAME="llm-providers"
REPO_SLUG="hibuka-labs/llm-providers"      # owner/name
DEFAULT_BRANCH="master"

# --- crates, in publish order ----------------------------------------------
# Published sequentially, so list dependencies before dependents.
# Format: "<crate-name>:<path-to-its-Cargo.toml>"
CRATES=(
  "llm-trait:llm-trait/Cargo.toml"
  "llm-unified:Cargo.toml"
)

# When true, all crates share one version number and bump together.
VERSION_LOCKSTEP=true

# Path deps that must carry a `version` requirement, or `cargo publish` refuses:
#   "dependent-toml|dependency-name|required-version"
INTERNAL_DEPS=(
  "Cargo.toml|llm-trait"
)

# --- gates ------------------------------------------------------------------
MSRV="1.88"
RUN_DOC_GATE=true          # cargo doc with -D warnings (catches bad intra-doc links)
RUN_FUZZ_BUILD=true        # compile fuzz targets, does not run them
FUZZ_SECONDS_ON_RELEASE=0  # >0 runs real fuzzing before publishing; slow, off by default

# --- CI ---------------------------------------------------------------------
CI_WORKFLOWS=("CI")        # workflow names that must be green
CI_POLL_SECONDS=20
CI_TIMEOUT_SECONDS=1800

# --- LLM (optional) --------------------------------------------------------
# Uses its own variable names on purpose: LLM_API_KEY / LLM_MODEL /
# LLM_BASE_URL are the environment contract of the library we are shipping, and
# are also read by llm-cli and examples/quickstart. Sourcing this repo's .env
# into those names for tooling would make an accidental `cargo run --bin
# llm-cli` talk to the wrong provider with the wrong key.
# Resolution order per setting: DEPLOY_LLM_*, then the LLM_* fallback below.
LLM_FALLBACK_VARS=true     # allow falling back to ambient LLM_API_KEY etc.
LLM_MODEL_FALLBACK="qwen3.8-flash"
LLM_TIMEOUT=90
LLM_REQUIRED=false         # true => hard-fail if no LLM is reachable
                           # false => use deterministic conventional-commit rules

# --- confirmation policy ---------------------------------------------------
# This is the knob for "mature it later". Right now the scripts are new and
# publishing to crates.io is irreversible, so confirm the irreversible steps.
#
#   interactive  : ask before each irreversible action (default, safest)
#   auto         : no prompts; requires --yes on the command line to do anything
#                  irreversible (use once the flow has been proven a few times)
CONFIRM_MODE="interactive"

# Steps that always require confirmation in interactive mode.
IRREVERSIBLE_STEPS=("version" "publish" "release")

# --- step order ------------------------------------------------------------
# Iterated in this sequence by release.sh. Your requested order is the default.
#
#   preflight  checks tools/auth/tree cleanliness, never writes anything
#   gates      fmt, clippy, tests, docs, (fuzz build)
#   version    LLM proposes the bump from online versions + git log; you confirm
#   commit     commit the bump locally, before anything is published
#   publish    cargo publish, sequentially, then confirm crates.io agrees
#   push       push the commit that was published
#   ci         wait for the workflows to go green
#   tag        create and push v<version>
#   release    create the GitHub release (LLM drafts the notes, you confirm)
#   verify     fresh throwaway project that depends only on the published versions
#
# Why `commit` comes before `publish`: publishing from a dirty tree lets
# crates.io hold bytes that no commit ever captured, which is unrecoverable if
# the run dies on the next line. Committing first makes every published artifact
# reproducible from pushed history.
#
# Publishing still precedes CI, and that has a real cost: a version can be
# burned by a commit that CI later rejects. Local gates run first, so this needs
# a CI-only failure (flaky runner, or something nightly-specific stable misses).
# To never risk it, reorder to:
#   preflight gates version commit push ci publish verify tag release
STEPS=("preflight" "gates" "version" "commit" "publish" "push" "ci" "tag" "release" "verify")

# --- layout -----------------------------------------------------------------
STATE_DIR="deploy/.state"
LOG_DIR="deploy/logs"
