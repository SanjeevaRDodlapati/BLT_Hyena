#!/bin/bash

echo "🔐 GitHub SSH Setup Instructions for BLT_Hyena Project"
echo "======================================================="
echo ""
echo "You need to add SSH keys to each GitHub account:"
echo ""

echo "1️⃣  For sdodlapati3 account (sdodlapati3@gatech.edu):"
echo "   → Open: https://github.com/settings/ssh"
echo "   → Click 'New SSH key'"
echo "   → Title: 'BLT_Hyena Development Key'"
echo "   → Key Type: Authentication Key"
echo "   → Copy and paste this key:"
echo ""
cat ~/.ssh/id_ed25519_sdodlapati3.pub
echo ""
echo "   → Click 'Add SSH key'"
echo ""

echo "2️⃣  For SanjeevaRDodlapati account (sdodl001@odu.edu):"
echo "   → Login to GitHub as SanjeevaRDodlapati"
echo "   → Open: https://github.com/settings/ssh"
echo "   → Click 'New SSH key'"
echo "   → Title: 'BLT_Hyena Development Key'"
echo "   → Key Type: Authentication Key"
echo "   → Copy and paste this key:"
echo ""
cat ~/.ssh/id_ed25519_sanjeevar.pub
echo ""
echo "   → Click 'Add SSH key'"
echo ""

echo "3️⃣  For sdodlapa account (sdodlapa@gmail.com):"
echo "   → Login to GitHub as sdodlapa"
echo "   → Open: https://github.com/settings/ssh"
echo "   → Click 'New SSH key'"
echo "   → Title: 'BLT_Hyena Development Key'"
echo "   → Key Type: Authentication Key"
echo "   → Copy and paste this key:"
echo ""
cat ~/.ssh/id_ed25519_sdodlapa.pub
echo ""
echo "   → Click 'Add SSH key'"
echo ""

echo "🧪 After adding all keys, run this script with 'test' to verify:"
echo "   ./setup_github_ssh.sh test"
echo ""

if [ "$1" = "test" ]; then
    echo "🧪 Testing SSH connections..."
    echo ""
    
    echo "Testing sdodlapati3..."
    ssh -T git@github.com-sdodlapati3 2>&1 | head -1
    
    echo "Testing sanjeevar..."
    ssh -T git@github.com-sanjeevar 2>&1 | head -1
    
    echo "Testing sdodlapa..."
    ssh -T git@github.com-sdodlapa 2>&1 | head -1
    
    echo ""
    echo "✅ If you see 'Hi [username]! You've successfully authenticated', the setup worked!"
    echo "❌ If you see 'Permission denied', add the SSH key to that GitHub account"
fi

if [ "$1" = "push" ]; then
    echo "🚀 Pushing code quality improvements to all repositories..."
    echo ""
    
    echo "Pushing to sdodlapati3..."
    git push sdodlapati3 main
    
    echo ""
    echo "Pushing to sanjeevar..."
    git push sanjeevar main
    
    echo ""
    echo "Pushing to sdodlapa..."
    git push sdodlapa main
    
    echo ""
    echo "✅ Push complete to all repositories!"
fi
