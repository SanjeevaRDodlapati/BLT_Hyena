# Git Scripts Directory

Simple and clean git management for repositories with multiple GitHub remotes.

## Main Script: `git_manager.sh`

**Two commands, zero confusion.** Just what you need, nothing more.

### Usage

```bash
./scripts/git_manager.sh [COMMAND] [MESSAGE]
```

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `sync` | Check if local repo is up to date with all remotes | `./scripts/git_manager.sh sync` |
| `commit` | Commit all changes and push to all remotes | `./scripts/git_manager.sh commit "Add new feature"` |

### Examples

```bash
# Check if your repo is synced (answers "is my repo up to date?")
./scripts/git_manager.sh sync

# Commit and push all changes to all remotes
./scripts/git_manager.sh commit "Fix authentication bug"

# Get help
./scripts/git_manager.sh help
```

## Configuration

- **Primary Remote**: `sdodlapati3` 
- **All Remotes**: Used for sync checks and pushes
- **Current Branch**: Automatically detected and used

## Benefits

✅ **Simple** - Only two commands to remember  
✅ **Clear** - No cryptic flags or confusing options  
✅ **Safe** - Always shows what's happening  
✅ **Smart** - Works with your current branch automatically  
✅ **Colorful** - Easy to read output with status indicators  

## Migration from Old Scripts

| Old Command | New Command |
|-------------|-------------|
| `./scripts/git_check_sync_status.sh` | `./scripts/git_manager.sh sync` |
| `./scripts/git_push_all_remotes.sh main` | Use `commit` after making changes |

## Backward Compatibility

The old `git_push_all_remotes.sh` script still works but redirects to the new system.
