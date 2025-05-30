# Session Notes Directory

This directory contains development session notes for the Hyena-GLT framework project.

## Purpose

Session notes provide detailed tracking of development progress, decisions made, issues encountered, and context for future sessions. They serve as a complement to the master `PROJECT_STATE.md` document.

## File Naming Convention

Session notes should be named using the following format:
```
session_notes_YYYY-MM-DD-XXX.md
```

Where:
- `YYYY-MM-DD` is the session date
- `XXX` is a sequential session number for that day (001, 002, etc.)

Examples:
- `session_notes_2025-01-28-001.md`
- `session_notes_2025-01-28-002.md`
- `session_notes_2025-01-29-001.md`

## Using the Template

1. Copy the template from `docs/SESSION_NOTES_TEMPLATE.md`
2. Rename it following the naming convention above
3. Fill in the sections as you work through your development session
4. Save the completed notes at the end of your session

## Template Location

The session notes template is located at:
```
docs/SESSION_NOTES_TEMPLATE.md
```

## Integration with Project State

- Session notes provide detailed, granular tracking
- `PROJECT_STATE.md` provides high-level status and context recovery
- Update both documents to maintain comprehensive project history

## Quick Commands

```bash
# Create new session notes from template
cp docs/SESSION_NOTES_TEMPLATE.md session_notes/session_notes_$(date +%Y-%m-%d)-001.md

# View recent session notes
ls -la session_notes/ | tail -5

# Find session notes by date
ls session_notes/session_notes_2025-01-28-*
```

## Best Practices

1. **Start Each Session**: Begin by creating session notes from the template
2. **Update Regularly**: Fill in sections as you work, don't wait until the end
3. **Be Specific**: Include file names, command outputs, and specific decisions
4. **Link Context**: Reference previous sessions and relate to PROJECT_STATE.md
5. **End Clean**: Complete all sections before ending the session

## Archive Policy

- Keep all session notes for project history
- Consider organizing into yearly subdirectories for long-running projects
- Session notes complement git history with decision context and reasoning

---

*This directory is part of the comprehensive state documentation system for efficient context recovery and knowledge transfer.*
