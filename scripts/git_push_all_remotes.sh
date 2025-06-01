#!/bin/bash
# DEPRECATED: Use ./scripts/git_manager.sh push instead
# This script is maintained for backward compatibility
# 
# Migration:
#   ./scripts/git_push_all_remotes.sh main       -> ./scripts/git_manager.sh push main
#   ./scripts/git_push_all_remotes.sh main --force-push -> ./scripts/git_manager.sh push main --force

echo "⚠️  DEPRECATED: This script is deprecated."
echo "💡 Use the new unified git manager instead:"
echo "   ./scripts/git_manager.sh push $1 ${2:+--force}"
echo ""
echo "🔄 Redirecting to new script..."
echo ""

# Parse arguments for compatibility
BRANCH=${1:-main}
FORCE_FLAG=""
if [ "$2" = "--force-push" ]; then
    FORCE_FLAG="--force"
fi

# Redirect to new script
exec "$(dirname "${BASH_SOURCE[0]}")/git_manager.sh" push "$BRANCH" $FORCE_FLAG

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
