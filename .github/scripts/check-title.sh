#!/usr/bin/env bash
# Checks a title against the project format: "<icon> type(scope): description", where the icon is the one
# that belongs to the type (see CONTRIBUTING.md, "Titles"). One script serves two callers so they cannot drift:
#
#   TITLE='✨ feat(cli): add x' bash .github/scripts/check-title.sh    the PR title check in CI
#   bash .github/scripts/check-title.sh --file <message-file>          the commit-msg hook (pre-commit)
#
# The title is read from the environment or a file, never from an argument, so it is never parsed as code.
# It avoids associative arrays so it also runs on the bash 3.2 that macOS ships.

set -u

commit_message=0
if [[ "${1:-}" == "--file" ]]; then
  if [[ -z "${2:-}" || ! -r "$2" ]]; then
    echo "check-title.sh: --file needs a readable commit message file" >&2
    exit 2
  fi
  IFS= read -r TITLE <"$2" || true # a last line without a newline still fills TITLE
  TITLE="${TITLE%$'\r'}"
  commit_message=1
fi

TITLE="${TITLE-}"

# Messages git writes itself while merging, or that "git rebase --autosquash" folds away.
if [[ "$commit_message" == 1 && "$TITLE" =~ ^(Merge |fixup!\ |squash!\ |amend!\ ) ]]; then
  exit 0
fi

icon_for() {
  case "$1" in
    feat) printf '✨' ;;
    fix) printf '🐛' ;;
    refactor) printf '♻' ;;
    perf) printf '⚡' ;;
    docs) printf '📝' ;;
    test) printf '✅' ;;
    build) printf '📦' ;;
    ci) printf '👷' ;;
    chore) printf '🔧' ;;
  esac
}

# Emoji such as ⚡️ and ♻️ may carry the invisible variation selector U+FE0F; ignore it.
vs=$(printf '\357\270\217')
title="${TITLE//$vs/}"

conventional='^([^[:space:]]+) (feat|fix|refactor|perf|docs|test|build|ci|chore)(\([a-z0-9._/-]+\))?!?: [^[:space:]]'
# GitHub's own "Revert" button titles the PR: Revert "✨ feat: add x"
revert='^Revert ".+"$'

if [[ "$title" =~ $revert ]]; then
  echo "Title OK: $TITLE"
  exit 0
fi
if [[ "$title" =~ $conventional ]]; then
  icon="${BASH_REMATCH[1]}"
  type="${BASH_REMATCH[2]}"
  if [[ "$icon" == "$(icon_for "$type")" ]]; then
    echo "Title OK: $TITLE"
    exit 0
  fi
fi

message="The title must look like '<icon> type(scope): description' with the icon that belongs to the type. Icons: ✨ feat, 🐛 fix, ♻️ refactor, ⚡️ perf, 📝 docs, ✅ test, 📦 build, 👷 ci, 🔧 chore. Example: 🐛 fix(cli): keep the config file when the input is invalid. Your title: $TITLE"
if [[ -n "${GITHUB_ACTIONS-}" ]]; then
  echo "::error::$message"
else
  echo "$message" >&2
fi
exit 1
