#!/bin/bash
# Quick GitHub Push Script
# Run this after creating your GitHub repository

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           SCBA - GitHub Push Instructions                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "BEFORE RUNNING THIS SCRIPT:"
echo "1. Create repository at: https://github.com/new"
echo "2. Name it: SCBA"
echo "3. Set to PUBLIC"
echo "4. Do NOT initialize with README"
echo ""
echo "Then update the USERNAME below and run this script"
echo ""

# UPDATE THIS LINE with your GitHub username
GITHUB_USERNAME="YOUR_USERNAME_HERE"

if [ "$GITHUB_USERNAME" == "YOUR_USERNAME_HERE" ]; then
    echo "❌ ERROR: Please update GITHUB_USERNAME in this script first!"
    echo ""
    echo "Edit this file and change:"
    echo "  GITHUB_USERNAME=\"YOUR_USERNAME_HERE\""
    echo "to:"
    echo "  GITHUB_USERNAME=\"your-actual-github-username\""
    exit 1
fi

echo "Will push to: https://github.com/$GITHUB_USERNAME/SCBA"
echo ""
read -p "Is this correct? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Step 1: Adding remote..."
    git remote add origin "https://github.com/$GITHUB_USERNAME/SCBA.git"
    
    echo "Step 2: Pushing to GitHub..."
    git push -u origin main
    
    echo ""
    echo "✅ Done! Your repository is now at:"
    echo "   https://github.com/$GITHUB_USERNAME/SCBA"
    echo ""
    echo "Next steps:"
    echo "  1. Visit the URL above to verify"
    echo "  2. Add repository topics (medical-imaging, explainable-ai, chest-xray)"
    echo "  3. Share link with supervisor/colleagues"
else
    echo ""
    echo "❌ Cancelled. Please update GITHUB_USERNAME and try again."
fi
