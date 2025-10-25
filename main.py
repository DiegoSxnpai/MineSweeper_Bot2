#!/usr/bin/env python3
"""
Screen-Based Minesweeper Solver
Automatically finds and solves real Minesweeper games on your screen.
"""

import sys
import argparse


def main():
    """Main entry point - directly start screen-based solving."""
    parser = argparse.ArgumentParser(description="Screen-Based Minesweeper Solver")
    parser.add_argument('--delay', '-d', type=float, default=1.5,
                       help='Delay between moves in seconds (default: 1.5)')
    parser.add_argument('--window', type=str, default=None,
                       help='Specific window title to solve (optional)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')

    args = parser.parse_args()

    print("Screen-Based Minesweeper Solver")
    print("=" * 50)
    print("This program will:")
    print("1. Find your Minesweeper game on screen")
    print("2. Read the board using computer vision")
    print("3. Solve it automatically by moving your mouse")
    print("=" * 50)

    try:
        from screen_solver import ScreenMinesweeperSolver

        if args.debug:
            print("Debug mode enabled")

        solver = ScreenMinesweeperSolver()

        if args.window:
            print(f"Targeting window: {args.window}")

        success = solver.find_and_solve(args.delay)

        if success:
            print("Successfully solved the Minesweeper game!")
        else:
            print("Could not solve the game. Make sure:")
            print("   - Minesweeper is visible on screen")
            print("   - The game window is not minimized")
            print("   - No other windows are covering the game")
            print("   - Try increasing the delay with --delay 2.0")
            print("   - Try --debug for more detailed output")

    except ImportError:
        print("Required libraries not installed!")
        print("\nTo install dependencies, run:")
        print("pip install opencv-python easyocr pyautogui pillow numpy")
        print("\nThen run: python main.py")
        return 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure your Minesweeper game is visible and try again.")
        print("Try using --debug for more detailed error information.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
