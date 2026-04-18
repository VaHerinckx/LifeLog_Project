#!/bin/zsh

# ============================================================================
# LifeLog Processing Launcher
# Double-click this file to start the LifeLog data processing pipeline
# ============================================================================

# Navigate to the project directory
cd "$(dirname "$0")"

# Initialize pyenv (needed when launched from Finder)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Activate the correct Python environment
pyenv activate general_coding

echo "=================================================="
echo "  LifeLog Processing Pipeline"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo ""

# Run the processing script (quiet mode by default, override with --verbose or --debug)
python src/process_exports.py --quiet "$@"

echo ""
echo "=================================================="
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo ""
echo "Press any key to close this window..."
read -k 1
