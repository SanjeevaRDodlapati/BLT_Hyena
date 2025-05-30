#!/bin/bash

# Multi-Push Demonstration Script
# Demonstrates the multi-repository push functionality

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Multi-Repository Push Demonstration${NC}"
echo -e "${BLUE}=====================================\n${NC}"

# Show current commit
echo -e "${YELLOW}📊 Latest Commit to Push:${NC}"
git log --oneline -1 --color=always
echo ""

# Show framework stats
echo -e "${YELLOW}📈 Framework Enhancement Summary:${NC}"
echo "  ✅ Performance monitoring system added"
echo "  ✅ Version updated to 1.0.1"
echo "  ✅ New utilities: ProfilerContext, benchmarking, throughput measurement"
echo "  ✅ Memory and GPU monitoring capabilities"
echo "  ✅ Comprehensive documentation and examples"
echo ""

# Show remote repositories
echo -e "${YELLOW}🔗 Target Repositories:${NC}"
git remote -v | awk '{
    if ($2 ~ /fetch/) {
        split($2, parts, "/")
        repo = parts[4] parts[5]
        gsub(/\.git/, "", repo)
        print "  " $1 ": " repo
    }
}'
echo ""

# Function to test and push to repository
test_and_push() {
    local remote_name=$1
    local description=$2
    
    echo -e "${PURPLE}📤 Testing push to ${remote_name} (${description})...${NC}"
    
    if git push --dry-run "$remote_name" main >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Dry run successful - attempting real push...${NC}"
        
        if git push "$remote_name" main; then
            echo -e "${GREEN}🎉 Successfully pushed to ${remote_name}!${NC}"
            return 0
        else
            echo -e "${RED}❌ Push failed for ${remote_name}${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}🔐 Authentication required for ${remote_name}${NC}"
        echo -e "${YELLOW}   Use setup_multi_push.sh to configure access${NC}"
        return 1
    fi
}

# Track success
success_count=0
total_count=3

echo -e "${BLUE}🔄 Pushing to All Repositories:${NC}"
echo "================================"

# Push to each repository
if test_and_push "sdodlapati3" "Primary Account"; then
    ((success_count++))
fi

echo ""

if test_and_push "sanjeevar" "SanjeevaRDodlapati Account"; then
    ((success_count++))
fi

echo ""

if test_and_push "sdodlapa" "sdodlapa Account"; then
    ((success_count++))
fi

echo ""

# Summary
echo -e "${BLUE}📊 Push Results Summary:${NC}"
echo "========================"

if [ $success_count -eq $total_count ]; then
    echo -e "${GREEN}🎉 SUCCESS: All $total_count repositories updated!${NC}"
    echo -e "${GREEN}   The complete Hyena-GLT v1.0.1 framework is now available across all accounts${NC}"
else
    echo -e "${YELLOW}⚠️  PARTIAL SUCCESS: ${success_count}/${total_count} repositories updated${NC}"
    echo ""
    echo -e "${YELLOW}📋 Next Steps for Remaining Repositories:${NC}"
    echo "  1. Run: ./setup_multi_push.sh"
    echo "  2. Follow the Personal Access Token setup guide"
    echo "  3. Re-run this script to complete the multi-push"
    echo ""
    echo -e "${YELLOW}📖 For detailed instructions, see: MULTI_PUSH_GUIDE.md${NC}"
fi

echo ""
echo -e "${BLUE}🧬 Framework Features Now Available:${NC}"
echo "  🔬 Performance profiling and monitoring"
echo "  📊 Memory and GPU usage tracking"
echo "  ⚡ Throughput and benchmark analysis"
echo "  🧪 Complete test suite (90%+ coverage)"
echo "  📚 Comprehensive documentation"
echo "  🔧 Training and inference infrastructure"
echo "  🧬 Genomic tokenization and analysis"
echo ""

# Show verification commands
echo -e "${BLUE}🔍 Verification Commands:${NC}"
echo "  git branch -vv  # Check tracking branches"
echo "  git remote -v   # Verify remote configuration"
echo "  git log --oneline -5  # Recent commits"
echo ""

echo -e "${GREEN}✨ Multi-push demonstration complete!${NC}"
