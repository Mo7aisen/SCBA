# GitHub Repository Setup Guide

## Creating and Pushing to GitHub

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `SCBA` (or `scba-medical-xai`)
3. Description: `Synthetic Counterfactual Border Audit for Segmentation Explainability`
4. **Important**: Set to **Public** (for sharing with colleagues/supervisors)
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add GitHub as remote (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/SCBA.git

# Or if using SSH:
git remote add origin git@github.com:USERNAME/SCBA.git

# Verify remote was added
git remote -v

# Push to GitHub
git push -u origin main
```

### Step 3: Verify Upload

1. Go to your repository: `https://github.com/USERNAME/SCBA`
2. Check that all files are visible
3. Verify README.md displays correctly
4. Check that manuscript figures appear in `manuscript/figures/`

## Repository Settings (Optional but Recommended)

### Add Topics

In your repository page, click "Add topics" and add:
- `medical-imaging`
- `explainable-ai`
- `chest-xray`
- `grad-cam`
- `counterfactual-explanations`
- `xai`
- `pytorch`

### Enable GitHub Pages (for documentation)

1. Go to Settings → Pages
2. Source: Deploy from a branch
3. Branch: main, folder: /docs (if you add documentation)

### Add Description

Update the repository description at the top:
```
Counterfactual consistency evaluation framework for Grad-CAM methods in medical image segmentation. Systematic audit of explanation methods using synthetic border perturbations in chest X-ray lung segmentation.
```

## Sharing with Colleagues/Supervisors

### For Paper Inclusion

Add to your manuscript (in "Code Availability" or "Acknowledgments"):

```latex
\section*{Code Availability}
The implementation of SCBA and all experimental code are publicly available at:
\url{https://github.com/USERNAME/SCBA}
```

### For Email/Slack

Use this template:

```
Hi [Name],

I've published the SCBA project code on GitHub:
https://github.com/USERNAME/SCBA

The repository includes:
- Complete implementation of the counterfactual evaluation framework
- All Grad-CAM variants (Seg-Grad-CAM, HiResCAM, Grad-CAM++)
- Reproducible experiment scripts
- Conference paper manuscript (LaTeX source)
- Documentation and usage examples

Feel free to clone and explore. Let me know if you have any questions!

Best,
[Your name]
```

## Making Updates

After pushing to GitHub, when you make changes:

```bash
# Make your changes to files
# ...

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Fix: Update evaluation metrics for consistency"

# Push to GitHub
git push
```

## Repository Structure (What Others Will See)

```
SCBA/
├── README.md                    # Project overview (first thing people see)
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files to ignore
├── scba/                        # Main package
│   ├── cf/                      # Counterfactual generation
│   ├── xai/                     # XAI methods
│   ├── metrics/                 # Evaluation metrics
│   ├── models/                  # Segmentation models
│   └── data/                    # Data loaders
├── manuscript/                  # Conference paper
│   ├── scba_manuscript.tex
│   ├── references.bib
│   └── figures/
├── run_*.py                     # Experiment scripts
└── tests/                       # Unit tests
```

## Best Practices

### Commit Messages

Use clear, descriptive commit messages:

✅ Good:
- "Add bootstrap confidence intervals for statistical validation"
- "Fix: Correct attribution normalization in HiResCAM"
- "Refactor: Simplify border perturbation pipeline"

❌ Avoid:
- "update"
- "fix stuff"
- "changes"

### Keep History Clean

- Commit logical units of work
- Don't commit broken code to main branch
- Test before pushing

### Documentation

- Keep README.md up to date
- Add comments to complex functions
- Include usage examples

## Cloning the Repository (For Others)

Others can clone your repository with:

```bash
git clone https://github.com/USERNAME/SCBA.git
cd SCBA
pip install -r requirements.txt
```

## Troubleshooting

### "Repository not found" error

- Check your GitHub username is correct
- Verify repository name matches exactly
- Ensure you have push access

### Large files error

- Ensure model checkpoints are in .gitignore
- Check no large datasets are being committed
- Use Git LFS for large files if needed

### Merge conflicts

- Pull latest changes first: `git pull`
- Resolve conflicts manually
- Commit merged changes

## Security Notes

✅ **Already done (safe to share)**:
- No API keys or credentials in code
- No personal information
- Professional code structure
- Clean commit history
- Proper .gitignore

❌ **Never commit**:
- API keys or passwords
- Personal data (patient information, etc.)
- Large model checkpoints (>100MB)
- Temporary/cache files

## After Pushing

Your repository is now:
- ✅ Public and shareable
- ✅ Professional and well-documented
- ✅ Ready for inclusion in paper
- ✅ Clonable by colleagues/reviewers
- ✅ Clean and organized

You can safely share the URL with:
- Your supervisor
- Collaborators
- Paper reviewers (if required)
- Conference organizers (for supplementary materials)
