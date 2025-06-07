# BLT_Hyena Tutorial Restructure Plan

## Current Problems

### Documentation Chaos
- 40+ scattered docs in `/docs/` folder
- Multiple overlapping "getting started" guides
- Over-complex cross-referencing system
- No clear learning progression
- Framework confusion (Hyena-GLT vs BLT core)

### User Experience Issues
- No single entry point for beginners
- Unclear which guide to follow for specific goals
- Too much cognitive overhead to find relevant information
- Advanced concepts mixed with basic tutorials

## Proposed Solution: Clean Tutorial System

### New Structure
```
BLT_Hyena/
├── tutorial/                    # NEW: Dedicated tutorial folder
│   ├── README.md               # Tutorial hub - single entry point
│   ├── 00_QUICK_START.md       # 5-minute demo
│   ├── 01_FUNDAMENTALS.md      # Core BLT concepts
│   ├── 02_HYENA_INTEGRATION.md # Hyena-specific features
│   ├── 03_DATA_PIPELINE.md     # Data processing
│   ├── 04_TRAINING.md          # Training workflows
│   ├── 05_EVALUATION.md        # Model evaluation
│   ├── 06_PRODUCTION.md        # Production deployment
│   ├── 07_ADVANCED.md          # Advanced topics
│   ├── examples/               # Working code examples
│   ├── assets/                 # Images, diagrams
│   └── troubleshooting/        # Common issues
└── docs/                       # Technical reference (keep existing)
```

### Learning Paths
1. **Quick Demo** → 00_QUICK_START.md (5 min)
2. **Beginner Path** → 01, 02, 03, 04 (2-3 hours)
3. **Developer Path** → 01, 03, 04, 05, 07 (4-5 hours)
4. **Production Path** → 01, 04, 05, 06 (3-4 hours)

### Key Principles
- **Progressive Disclosure**: Start simple, add complexity gradually
- **Self-Contained**: Each tutorial works independently
- **Practical Focus**: Every tutorial includes working code
- **Clear Prerequisites**: Explicit requirements for each section
- **Consistent Structure**: Same format across all tutorials

## Implementation Plan

### Phase 1: Foundation (This session)
- Create tutorial folder structure
- Write 00_QUICK_START.md with working example
- Create tutorial README.md as single entry point
- Test with basic example

### Phase 2: Core Tutorials
- 01_FUNDAMENTALS.md - BLT architecture basics
- 02_HYENA_INTEGRATION.md - Hyena-specific features
- 03_DATA_PIPELINE.md - Data processing workflows

### Phase 3: Advanced Content
- 04_TRAINING.md - Training strategies
- 05_EVALUATION.md - Evaluation methods
- 06_PRODUCTION.md - Deployment guide
- 07_ADVANCED.md - Advanced topics

### Phase 4: Polish
- Add working examples for each tutorial
- Create troubleshooting guide
- Add visual diagrams
- User testing and feedback

## Benefits

### For New Users
- Single entry point eliminates confusion
- Progressive learning path
- Working examples in every tutorial
- Clear next steps

### For Existing Docs
- Keep technical references in `/docs/`
- Add clear links between tutorial and reference
- Reduce redundancy
- Maintain API documentation

### For Maintainers
- Easier to update and maintain
- Clear separation of concerns
- Better user feedback loop
- Reduced support burden

## Success Metrics
- Time to first working example < 5 minutes
- Tutorial completion rate > 80%
- Reduced documentation-related issues
- Positive user feedback

## Next Steps
1. Create basic tutorial structure
2. Write and test quick start guide
3. Gradually migrate best content from existing docs
4. Gather user feedback
5. Iterate and improve
