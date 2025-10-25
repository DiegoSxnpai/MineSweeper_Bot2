#!/bin/bash
echo "Installing Screen-Based Minesweeper Solver dependencies..."
echo "This may take a few minutes..."
echo ""

pip install opencv-python easyocr pyautogui pillow numpy

echo ""
echo "Installation complete!"
echo ""
echo "To use the solver:"
echo "1. Open your Minesweeper game"
echo "2. Run: python main.py"
echo ""
echo "The program will automatically find and solve your Minesweeper game!"
echo ""
echo "For help: python main.py --help"
