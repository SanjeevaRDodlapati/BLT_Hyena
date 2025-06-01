#!/bin/bash
# Simple Git Multi-Remote Manager
# Two main commands for multi-remote git operations
#
# Usage: ./scripts/git_manager.sh [command] [message]
#
# Commands:
#   sync      - Sync local repo with all remotes (fetch and check status)
#   commit    - Commit all changes and push to all remotes
#
# Examples:
#   ./scripts/git_manager.sh sync
#   ./scripts/git_manager.sh commit "Add new feature"

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PRIMARY_REMOTE="sdodlapati3"  # Your primary remote

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Helper functions
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_header() { echo -e "${PURPLE}🚀 $1${NC}"; }

show_help() {
    cat << EOF
🛠️  Simple Git Multi-Remote Manager

USAGE:
    ./scripts/git_manager.sh [COMMAND] [MESSAGE]

COMMANDS:
    sync      Check sync status with all remotes and fetch latest changes
    commit    Commit all changes and push to all configured remotes

EXAMPLES:
    ./scripts/git_manager.sh sync
    ./scripts/git_manager.sh commit "Add new feature"
    ./scripts/git_manager.sh commit "Fix bug in authentication"

REMOTES:
    Primary: $PRIMARY_REMOTE (used for pulls)
    All configured remotes will be used for pushes
EOF
}

sync_with_remotes() {
    log_header "Syncing with All Remotes"
    echo "🌿 Current branch: $(git branch --show-current)"
    echo ""
    
    cd "$REPO_DIR" || exit 1
    
    # Fetch from all remotes
    log_info "Fetching from all remotes..."
    if git fetch --all --prune; then
        log_success "Successfully fetched from all remotes"
    else
        log_error "Failed to fetch from some remotes"
        return 1
    fi
    echo ""
    
    local remotes=$(git remote)
    if [ -z "$remotes" ]; then
        log_error "No remote repositories configured"
        return 1
    fi
    
    echo "📡 Configured remotes:"
    git remote -v
    echo ""
    
    local current_branch=$(git branch --show-current)
    local all_synced=true
    
    echo "📊 Sync Status:"
    for remote in $remotes; do
        echo "🔗 Checking $remote..."
        
        if git show-ref --verify --quiet "refs/remotes/$remote/$current_branch"; then
            local ahead=$(git rev-list --count "$remote/$current_branch..HEAD" 2>/dev/null || echo "0")
            local behind=$(git rev-list --count "HEAD..$remote/$current_branch" 2>/dev/null || echo "0")
            
            if [ "$ahead" -eq 0 ] && [ "$behind" -eq 0 ]; then
                echo "   ✅ Up to date with $remote"
            elif [ "$ahead" -gt 0 ] && [ "$behind" -eq 0 ]; then
                echo "   ⬆️  Ahead by $ahead commits (ready to push)"
                all_synced=false
            elif [ "$ahead" -eq 0 ] && [ "$behind" -gt 0 ]; then
                echo "   ⬇️  Behind by $behind commits (need to pull)"
                all_synced=false
            else
                echo "   🔀 Diverged: $ahead ahead, $behind behind"
                all_synced=false
            fi
        else
            echo "   ❓ Remote branch doesn't exist"
            all_synced=false
        fi
        echo ""
    done
    
    if [ "$all_synced" = true ]; then
        log_success "Repository is perfectly synchronized!"
    else
        log_warning "Repository has differences with remotes"
        echo "💡 Use 'commit' command to push your changes"
    fi
}

commit_and_push() {
    local commit_message="$1"
    
    if [ -z "$commit_message" ]; then
        log_error "Commit message is required"
        echo "Usage: ./scripts/git_manager.sh commit \"Your commit message\""
        return 1
    fi
    
    log_header "Commit and Push to All Remotes"
    echo "📝 Message: $commit_message"
    echo ""
    
    cd "$REPO_DIR" || exit 1
    
    # Check if there are any changes
    if git diff-index --quiet HEAD --; then
        log_warning "No changes to commit"
        return 0
    fi
    
    # Show what will be committed
    echo "📊 Changes to be committed:"
    git status --short
    echo ""
    
    # Add all changes
    log_info "Adding all changes..."
    git add -A
    
    # Commit changes
    log_info "Committing changes..."
    if git commit -m "$commit_message"; then
        log_success "Successfully committed changes"
    else
        log_error "Failed to commit changes"
        return 1
    fi
    echo ""
    
    # Push to all remotes
    local remotes=$(git remote)
    if [ -z "$remotes" ]; then
        log_error "No remote repositories configured"
        return 1
    fi
    
    local current_branch=$(git branch --show-current)
    local success_count=0
    local total_count=0
    
    for remote in $remotes; do
        total_count=$((total_count + 1))
        echo "🔄 Pushing to $remote..."
        
        if git push "$remote" "$current_branch"; then
            log_success "Pushed to $remote"
            success_count=$((success_count + 1))
        else
            log_error "Failed to push to $remote"
        fi
        echo ""
    done
    
    echo "📊 Push Summary:"
    echo "   ✅ Successful: $success_count/$total_count"
    
    if [ $success_count -eq $total_count ]; then
        log_success "All remotes updated successfully!"
        return 0
    else
        log_warning "Some pushes failed"
        return 1
    fi
}

# Main script logic
main() {
    local command=${1:-help}
    
    case $command in
        sync)
            sync_with_remotes
            ;;
        commit)
            shift
            commit_and_push "$*"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
