#!/bin/bash
# Setup script for Image Caption Generation project

echo "🚀 Setting up Image Caption Generation Environment"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv kenv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source kenv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo "🔍 Verifying installation..."
python -c "import tensorflow as tf; print('✓ TensorFlow:', tf.__version__)" || echo "❌ TensorFlow installation failed"
python -c "import numpy as np; print('✓ NumPy:', np.__version__)" || echo "❌ NumPy installation failed"
python -c "from PIL import Image; print('✓ Pillow is working')" || echo "❌ Pillow installation failed"
python -c "import matplotlib; print('✓ Matplotlib:', matplotlib.__version__)" || echo "❌ Matplotlib installation failed"

echo ""
echo "🎉 Setup complete!"
echo "=================================================="
echo "To activate the environment in the future, run:"
echo "  source kenv/bin/activate"
echo ""
echo "To check project status:"
echo "  python status_check.py"
echo ""
echo "To start using the project:"
echo "  python demo_simple.py"
echo "=================================================="
