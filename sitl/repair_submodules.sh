#!/bin/bash
# Repair submodules left half-checked-out by an interrupted clone.
# A broken one contains only its .git pointer file (or nothing at all).
set -u
cd ~/PX4-Autopilot

echo "=== scanning for empty submodule checkouts ==="
broken=()
while read -r _hash path _rest; do
    n=$(ls -A "$path" 2>/dev/null | grep -cv '^\.git$' || true)
    if [ "${n:-0}" -eq 0 ]; then
        echo "EMPTY: $path"
        broken+=("$path")
    fi
done < <(git submodule status --recursive)

if [ ${#broken[@]} -eq 0 ]; then
    echo "no broken submodules found"
    exit 0
fi

echo "=== repairing ${#broken[@]} submodule(s) ==="
for p in "${broken[@]}"; do
    echo "--- $p"
    git submodule deinit -f "$p" 2>/dev/null || true
    rm -rf ".git/modules/$p"
    git submodule update --init --recursive --force "$p"
done

echo "=== rescan ==="
left=0
while read -r _hash path _rest; do
    n=$(ls -A "$path" 2>/dev/null | grep -cv '^\.git$' || true)
    if [ "${n:-0}" -eq 0 ]; then
        echo "STILL EMPTY: $path"
        left=$((left+1))
    fi
done < <(git submodule status --recursive)
echo "REPAIR_DONE remaining_broken=$left"
