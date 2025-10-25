import random
import time
from typing import List, Tuple, Set, Optional


class Minesweeper:
    """A Minesweeper game implementation."""

    def __init__(self, width: int = 16, height: int = 16, num_mines: int = 40):
        self.width = width
        self.height = height
        self.num_mines = num_mines

        # Game state
        self.board: List[List[int]] = []  # -1 = mine, 0-8 = number of adjacent mines
        self.revealed: List[List[bool]] = []  # Which cells are revealed
        self.flagged: List[List[bool]] = []  # Which cells are flagged as mines
        self.game_over = False
        self.won = False
        self.first_move = True

        self._initialize_board()

    def _initialize_board(self):
        """Initialize the game board."""
        # Initialize empty board
        self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.revealed = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.flagged = [[False for _ in range(self.width)] for _ in range(self.height)]

        # Place mines randomly (but not on first click position)
        self._place_mines()

        # Calculate numbers
        self._calculate_numbers()

    def _place_mines(self):
        """Place mines randomly on the board."""
        mines_placed = 0
        while mines_placed < self.num_mines:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)

            if self.board[y][x] != -1:  # Not already a mine
                self.board[y][x] = -1
                mines_placed += 1

    def _calculate_numbers(self):
        """Calculate the numbers showing adjacent mine counts."""
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] != -1:
                    # Count adjacent mines
                    count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                if self.board[ny][nx] == -1:
                                    count += 1
                    self.board[y][x] = count

    def reveal(self, x: int, y: int) -> bool:
        """Reveal a cell. Returns True if mine was hit."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        if self.revealed[y][x] or self.flagged[y][x]:
            return False

        if self.first_move:
            # Ensure first move is safe by moving mines if necessary
            self._ensure_safe_first_move(x, y)
            self.first_move = False

        self.revealed[y][x] = True

        if self.board[y][x] == -1:
            self.game_over = True
            return True  # Mine hit

        # Auto-reveal empty cells
        if self.board[y][x] == 0:
            self._reveal_empty_area(x, y)

        # Check for win condition
        self._check_win_condition()

        return False

    def _ensure_safe_first_move(self, x: int, y: int):
        """Ensure the first move is safe by moving mines if necessary."""
        if self.board[y][x] == -1:
            # Move the mine to a different location
            self.board[y][x] = 0

            # Find a new spot for the mine
            for ny in range(self.height):
                for nx in range(self.width):
                    if self.board[ny][nx] != -1 and (nx != x or ny != y):
                        self.board[ny][nx] = -1
                        break
                else:
                    continue
                break

            # Recalculate numbers
            self._calculate_numbers()

    def _reveal_empty_area(self, x: int, y: int):
        """Recursively reveal connected empty cells."""
        stack = [(x, y)]

        while stack:
            cx, cy = stack.pop()

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = cx + dx, cy + dy
                    if (0 <= nx < self.width and 0 <= ny < self.height and
                        not self.revealed[ny][nx] and not self.flagged[ny][nx] and
                        self.board[ny][nx] >= 0):
                        self.revealed[ny][nx] = True
                        if self.board[ny][nx] == 0:
                            stack.append((nx, ny))

    def flag(self, x: int, y: int):
        """Toggle flag on a cell."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        if not self.revealed[y][x]:
            self.flagged[y][x] = not self.flagged[y][x]

    def _check_win_condition(self):
        """Check if the game has been won."""
        for y in range(self.height):
            for x in range(self.width):
                if not self.revealed[y][x] and self.board[y][x] != -1:
                    return  # Still cells to reveal

        self.game_over = True
        self.won = True

    def get_visible_board(self) -> List[List[str]]:
        """Get the current visible state of the board."""
        visible = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if self.revealed[y][x]:
                    if self.board[y][x] == -1:
                        row.append('*')  # Mine (shown when game over)
                    else:
                        row.append(str(self.board[y][x]))
                elif self.flagged[y][x]:
                    row.append('F')
                else:
                    row.append('#')
            visible.append(row)
        return visible

    def print_board(self):
        """Print the current board state."""
        visible = self.get_visible_board()

        # Print column numbers
        col_nums = "  " + " ".join(f"{i:2d}" for i in range(self.width))
        print(col_nums)
        print("  " + "-" * (self.width * 3))

        # Print rows
        for y, row in enumerate(visible):
            row_str = " ".join(cell for cell in row)
            print(f"{y:2d}|{row_str}")

    def get_game_state(self) -> dict:
        """Get the current game state as a dictionary."""
        return {
            'board': [row[:] for row in self.board],
            'revealed': [row[:] for row in self.revealed],
            'flagged': [row[:] for row in self.flagged],
            'game_over': self.game_over,
            'won': self.won,
            'width': self.width,
            'height': self.height
        }
