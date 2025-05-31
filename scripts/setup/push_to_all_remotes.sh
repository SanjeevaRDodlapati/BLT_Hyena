#!/bin/bash

# Script to push Hyena-GLT framework to all three GitHub repositories
# Usage: ./push_to_all_remotes.sh

echo "🚀 Pushing Hyena-GLT Framework to Multiple GitHub Repositories"
echo "================================================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Repository information
REPOS=(
    "sdodlapati3:https://github.com/sdodlapati3/BLT_Hyena.git"
    "sanjeevar:https://github.com/SanjeevaRDodlapati/BLT_Hyena.git" 
    "sdodlapa:https://github.com/sdodlapa/BLT_Hyena.git"
)

# Function to push to a remote
push_to_remote() {
    local remote_name=$1
    local repo_url=$2
    
    echo -e "\n${YELLOW}📤 Attempting to push to ${remote_name}...${NC}"
    echo "Repository: $repo_url"
    
    # Try to push
    if git push -u "$remote_name" main; then
        echo -e "${GREEN}✅ Successfully pushed to ${remote_name}${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to push to ${remote_name}${NC}"
        echo -e "${YELLOW}💡 You may need to authenticate for this account${NC}"
        return 1
    fi
}

# Main execution
echo "Current repository status:"
git status --short
echo ""

echo "Current remotes:"
git remote -v
echo ""

echo "Recent commits to push:"
git log --oneline -3
echo ""

# Track success/failure
success_count=0
total_count=${#REPOS[@]}

# Push to each repository
for repo_info in "${REPOS[@]}"; do
    IFS=':' read -r remote_name repo_url <<< "$repo_info"
    
    if push_to_remote "$remote_name" "$repo_url"; then
        ((success_count++))
    fi
done

# Summary
echo -e "\n${YELLOW}📊 Push Summary:${NC}"
echo "Successfully pushed to: $success_count/$total_count repositories"

if [ $success_count -eq $total_count ]; then
    echo -e "${GREEN}🎉 All repositories updated successfully!${NC}"
else
    echo -e "\n${YELLOW}⚠️  Authentication Required:${NC}"
    echo "For repositories that failed, you need to authenticate with the respective GitHub accounts."
    echo ""
    echo "Options:"
    echo "1. Use Personal Access Tokens (recommended)"
    echo "2. Switch GitHub accounts in your browser and try git credential manager"
    echo "3. Use SSH keys if configured for each account"
    echo ""
    echo "To use Personal Access Tokens:"
    echo "1. Go to GitHub → Settings → Developer settings → Personal access tokens"
    echo "2. Generate tokens for each account with 'repo' permissions"
    echo "3. Update remotes with: git remote set-url <remote> https://[TOKEN]@github.com/[USER]/BLT_Hyena.git"
fi

echo -e "\n${GREEN}Framework Status: Complete Hyena-GLT implementation ready! 🧬${NC}"
