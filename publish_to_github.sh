#!/usr/bin/env bash
set -Eeuo pipefail
repo_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$repo_dir"
owner=jolaaa999
repository=HyperSynergyV3
if ! command -v gh >/dev/null 2>&1; then
    echo "请先安装 GitHub CLI (gh)，再运行此脚本。" >&2
    exit 1
fi
if ! gh auth status --hostname github.com >/dev/null 2>&1; then
    gh auth login --hostname github.com --git-protocol https --web
fi
login="$(gh api user --jq .login | tr '[:upper:]' '[:lower:]')"
if [[ "$login" != "$owner" ]]; then
    echo "当前 GitHub 账号为 $login，目标账号为 $owner；请切换账号后重试。" >&2
    exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "存在尚未提交的修改，请先确认并提交后再推送。" >&2
    exit 1
fi
if ! gh repo view "$owner/$repository" --json nameWithOwner >/dev/null 2>&1; then
    gh repo create "$owner/$repository" --private --description "HyperSynergy V3 with a biological graph cell encoder"
fi
remote="https://github.com/$owner/$repository.git"
if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$remote"
else
    git remote add origin "$remote"
fi
gh auth setup-git --hostname github.com
git push --set-upstream origin main
gh repo view "$owner/$repository" --json url --jq .url
