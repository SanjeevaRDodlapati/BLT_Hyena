#!/bin/bash
# Push to all configured remote repositories
# Usage: ./scripts/git_push_all_remotes.sh [branch_name]

BRANCH=${1:-main}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🚀 Pushing to all remote repositories..."
echo "📁 Repository: $REPO_DIR"
echo "🌿 Branch: $BRANCH"
echo ""

cd "$REPO_DIR" || exit 1

# Get list of all remotes
REMOTES=$(git remote)

if [ -z "$REMOTES" ]; then
    echo "❌ No remote repositories configured"
    exit 1
fi

echo "📡 Configured remotes:"
git remote -v
echo ""

# Push to each remote
SUCCESS_COUNT=0
TOTAL_COUNT=0

for remote in $REMOTES; do
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo "🔄 Pushing to $remote..."
    
    if git push "$remote" "$BRANCH"; then
        echo "✅ Successfully pushed to $remote"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ Failed to push to $remote"
    fi
    echo ""
done

echo "📊 Summary:"
echo "   ✅ Successful pushes: $SUCCESS_COUNT/$TOTAL_COUNT"

if [ $SUCCESS_COUNT -eq $TOTAL_COUNT ]; then
    echo "🎉 All remotes updated successfully!"
    exit 0
else
    echo "⚠️  Some pushes failed. Check the output above for details."
    exit 1
fi
