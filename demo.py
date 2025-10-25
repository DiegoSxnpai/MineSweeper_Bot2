#!/usr/bin/env python3
"""
Minesweeper Solver Demo
Quick demonstration of the automated solver capabilities.
"""

import time
from minesweeper import Minesweeper
from solver import MinesweeperSolver


def demo_basic_solving():
    """Demonstrate basic solving on a small board."""
    print("Demo: Basic Minesweeper Solving")
    print("=" * 40)

    # Create a small game for easy demonstration
    game = Minesweeper(width=9, height=9, num_mines=10)
    solver = MinesweeperSolver(game)

    print("Starting position:")
    game.print_board()
    print()

    # Make a few moves to demonstrate
    moves = [(4, 4), (2, 2), (6, 6)]  # Some safe starting moves

    for i, (x, y) in enumerate(moves, 1):
        print(f"Move {i}: Revealing ({x}, {y})")
        game.reveal(x, y)

        if game.game_over:
            print("💥 Hit a mine!")
            break

        game.print_board()
        print()

        # Let solver make some moves
        for j in range(3):  # Let solver make 3 moves
            move = solver.solve()
            if move:
                mx, my = move
                print(f"Solver move {j+1}: Revealing ({mx}, {my})")
                game.reveal(mx, my)

                if game.game_over:
                    print("💥 Solver hit a mine!")
                    break

                game.print_board()
                print()

            if game.game_over:
                break

        if game.game_over:
            break

        time.sleep(1)

    print(f"Demo ended. Game over: {game.game_over}, Won: {game.won}")


def demo_solver_statistics():
    """Show solver statistics on different board sizes."""
    print("\nDemo: Solver Performance Statistics")
    print("=" * 50)

    test_configs = [
        ("Beginner", 9, 9, 10),
        ("Intermediate", 16, 16, 40),
        ("Expert", 16, 30, 99),
    ]

    for name, width, height, mines in test_configs:
        print(f"\nTesting {name} difficulty ({width}x{height}, {mines} mines)")

        games_won = 0
        total_moves = 0

        for i in range(3):  # Test 3 games each
            game = Minesweeper(width, height, mines)
            solver = MinesweeperSolver(game)

            # Play without display for speed
            won = solver.play_game(delay=0)
            if won:
                games_won += 1
            total_moves += solver.moves_made

        win_rate = (games_won / 3) * 100
        avg_moves = total_moves / 3

        print(f"   Win rate: {win_rate:.1f}%")
        print(f"   Average moves: {avg_moves:.1f}")


def demo_interactive_session():
    """Interactive demo where user can choose parameters."""
    print("\nInteractive Demo Session")
    print("=" * 40)
    print("Choose your preferred difficulty:")

    difficulties = {
        '1': ("Beginner", 9, 9, 10),
        '2': ("Intermediate", 16, 16, 40),
        '3': ("Expert", 16, 30, 99),
        '4': ("Custom", 16, 16, 40),
    }

    for key, (name, w, h, m) in difficulties.items():
        print(f"   {key}. {name} ({w}x{h}, {m} mines)")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice in difficulties:
        name, width, height, num_mines = difficulties[choice]

        if choice == '4':
            # Custom settings
            try:
                width = int(input(f"Width (current {width}): ") or width)
                height = int(input(f"Height (current {height}): ") or height)
                num_mines = int(input(f"Mines (current {num_mines}): ") or num_mines)
            except ValueError:
                print("Using default custom settings...")

        print(f"\nStarting {name} game: {width}x{height} with {num_mines} mines")
        print("Watch the solver in action!\n")

        game = Minesweeper(width, height, num_mines)
        solver = MinesweeperSolver(game)
        solver.play_game(delay=0.3, max_moves=200)

        print(f"\nGame completed! Result: {'Won' if game.won else 'Lost'}")
        print(f"Moves made: {solver.moves_made}")


def main():
    """Run the demo suite."""
    print("Automated Minesweeper Solver - Demo Suite")
    print("=" * 60)
    print("This demo will show you the capabilities of the automated solver.")
    print()

    # Run basic solving demo
    demo_basic_solving()

    # Show statistics
    demo_solver_statistics()

    # Interactive session
    try:
        response = input("\nWould you like to try an interactive session? (y/n): ").strip().lower()
        if response.startswith('y'):
            demo_interactive_session()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("Run 'python main.py --help' for full usage options.")
    print("Try 'python main.py --interactive' for a guided experience.")


if __name__ == "__main__":
    main()
