#!/bin/bash

# Interactive GitHub Multi-Account Push Helper
# This script helps you authenticate and push to multiple GitHub accounts

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔐 GitHub Multi-Account Push Helper${NC}"
echo "======================================"

# Check current status
echo -e "\n${YELLOW}📊 Current Repository Status:${NC}"
echo "Repository: $(pwd)"
echo "Branch: $(git branch --show-current)"
echo "Latest commit: $(git log --oneline -1)"

echo -e "\n${YELLOW}🔗 Current Remotes:${NC}"
git remote -v

echo -e "\n${YELLOW}✅ Successfully Pushed To:${NC}"
if git ls-remote --exit-code sdodlapati3 >/dev/null 2>&1; then
    echo "- sdodlapati3/BLT_Hyena ✅"
else
    echo "- sdodlapati3/BLT_Hyena ❌"
fi

echo -e "\n${YELLOW}🔄 Pending Push To:${NC}"
echo "- SanjeevaRDodlapati/BLT_Hyena"
echo "- sdodlapa/BLT_Hyena"

# Function to test repository access
test_repo_access() {
    local remote_name=$1
    echo -e "\n${BLUE}🔍 Testing access to ${remote_name}...${NC}"
    
    if git ls-remote --exit-code "$remote_name" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Access granted to ${remote_name}${NC}"
        return 0
    else
        echo -e "${RED}❌ Access denied to ${remote_name}${NC}"
        return 1
    fi
}

# Function to setup token authentication
setup_token_auth() {
    local remote_name=$1
    local repo_url=$2
    
    echo -e "\n${YELLOW}🔑 Setting up token authentication for ${remote_name}${NC}"
    echo "1. Go to GitHub.com and log into the ${remote_name} account"
    echo "2. Navigate to: Settings → Developer settings → Personal access tokens → Tokens (classic)"
    echo "3. Click 'Generate new token (classic)'"
    echo "4. Give it a name like 'BLT_Hyena_Push'"
    echo "5. Select 'repo' scope (full control of private repositories)"
    echo "6. Click 'Generate token'"
    echo ""
    
    read -p "Have you generated the token? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Enter the personal access token for ${remote_name}:${NC}"
        read -s token
        echo ""
        
        # Extract username from repo URL
        username=$(echo "$repo_url" | sed 's|https://github.com/||' | sed 's|/.*||')
        
        # Update remote URL with token
        new_url="https://${token}@github.com/${username}/BLT_Hyena.git"
        git remote set-url "$remote_name" "$new_url"
        
        echo -e "${GREEN}✅ Token configured for ${remote_name}${NC}"
        
        # Test the new authentication
        if test_repo_access "$remote_name"; then
            echo -e "${GREEN}🚀 Ready to push to ${remote_name}${NC}"
            return 0
        else
            echo -e "${RED}❌ Token authentication failed${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⏸️  Skipping ${remote_name} for now${NC}"
        return 1
    fi
}

# Function to push to repository
push_to_repo() {
    local remote_name=$1
    
    echo -e "\n${BLUE}📤 Pushing to ${remote_name}...${NC}"
    
    if git push -u "$remote_name" main; then
        echo -e "${GREEN}✅ Successfully pushed to ${remote_name}${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to push to ${remote_name}${NC}"
        return 1
    fi
}

# Main execution
echo -e "\n${BLUE}🎯 Choose Authentication Method:${NC}"
echo "1. Personal Access Tokens (Recommended)"
echo "2. Test current access first"
echo "3. Skip authentication setup"

read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo -e "\n${BLUE}🔑 Setting up Personal Access Token Authentication${NC}"
        
        # Setup for sanjeevar
        if ! test_repo_access "sanjeevar"; then
            setup_token_auth "sanjeevar" "https://github.com/SanjeevaRDodlapati/BLT_Hyena.git"
        fi
        
        # Setup for sdodlapa  
        if ! test_repo_access "sdodlapa"; then
            setup_token_auth "sdodlapa" "https://github.com/sdodlapa/BLT_Hyena.git"
        fi
        ;;
    2)
        echo -e "\n${BLUE}🔍 Testing Current Access${NC}"
        test_repo_access "sanjeevar"
        test_repo_access "sdodlapa"
        ;;
    3)
        echo -e "\n${YELLOW}⏸️  Skipping authentication setup${NC}"
        ;;
esac

# Ask if ready to push
echo -e "\n${BLUE}🚀 Ready to push to repositories?${NC}"
read -p "Push to all accessible repositories? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    success_count=0
    
    # Try pushing to each repository
    if test_repo_access "sanjeevar" >/dev/null 2>&1; then
        if push_to_repo "sanjeevar"; then
            ((success_count++))
        fi
    else
        echo -e "${YELLOW}⏸️  Skipping sanjeevar (no access)${NC}"
    fi
    
    if test_repo_access "sdodlapa" >/dev/null 2>&1; then
        if push_to_repo "sdodlapa"; then
            ((success_count++))
        fi
    else
        echo -e "${YELLOW}⏸️  Skipping sdodlapa (no access)${NC}"
    fi
    
    # Final summary
    echo -e "\n${BLUE}📊 Final Summary:${NC}"
    echo "- sdodlapati3: ✅ Already pushed"
    
    if git ls-remote --exit-code sanjeevar >/dev/null 2>&1 && git merge-base --is-ancestor HEAD sanjeevar/main 2>/dev/null; then
        echo "- sanjeevar: ✅ Successfully pushed"
    else
        echo "- sanjeevar: ❌ Needs authentication"
    fi
    
    if git ls-remote --exit-code sdodlapa >/dev/null 2>&1 && git merge-base --is-ancestor HEAD sdodlapa/main 2>/dev/null; then
        echo "- sdodlapa: ✅ Successfully pushed"  
    else
        echo "- sdodlapa: ❌ Needs authentication"
    fi
    
    echo -e "\n${GREEN}🎉 Hyena-GLT Framework Push Complete!${NC}"
    echo "See MULTI_PUSH_GUIDE.md for detailed instructions."
fi
