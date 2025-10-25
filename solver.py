import random
import time
from typing import List, Tuple, Set, Optional, Dict
from collections import defaultdict


class MinesweeperSolver:
    """Automated Minesweeper solver using logical deduction."""

    def __init__(self, game):
        self.game = game
        self.moves_made = 0

    def solve(self) -> Optional[Tuple[int, int]]:
        """Find the next best move. Returns (x, y) or None if no safe move found."""
        # Try basic strategy first
        move = self._basic_strategy()
        if move:
            return move

        # Try advanced strategy
        move = self._advanced_strategy()
        if move:
            return move

        # If no safe move found, try educated guess
        return self._educated_guess()

    def _basic_strategy(self) -> Optional[Tuple[int, int]]:
        """Apply basic Minesweeper solving strategies."""
        # Strategy 1: If a revealed cell has a number equal to its flagged neighbors,
        # all remaining hidden neighbors are safe
        for y in range(self.game.height):
            for x in range(self.game.width):
                if (self.game.revealed[y][x] and
                    self.game.board[y][x] > 0 and
                    not self.game.game_over):

                    flagged_neighbors = self._count_flagged_neighbors(x, y)
                    hidden_neighbors = self._count_hidden_neighbors(x, y)

                    if flagged_neighbors == self.game.board[y][x]:
                        # All remaining hidden neighbors are safe
                        safe_moves = self._get_hidden_neighbors(x, y)
                        if safe_moves:
                            return random.choice(safe_moves)

        # Strategy 2: If a revealed cell has a number equal to its hidden neighbors,
        # all hidden neighbors are mines
        for y in range(self.game.height):
            for x in range(self.game.width):
                if (self.game.revealed[y][x] and
                    self.game.board[y][x] > 0 and
                    not self.game.game_over):

                    flagged_neighbors = self._count_flagged_neighbors(x, y)
                    hidden_neighbors = self._count_hidden_neighbors(x, y)

                    if (flagged_neighbors + hidden_neighbors == self.game.board[y][x] and
                        hidden_neighbors > 0):
                        # All remaining hidden neighbors are mines
                        mine_moves = self._get_hidden_neighbors(x, y)
                        if mine_moves:
                            # Flag one of them
                            mx, my = random.choice(mine_moves)
                            self.game.flag(mx, my)
                            self.moves_made += 1
                            return self.solve()  # Look for more moves

        # Strategy 3: Look for cells with 0 hidden neighbors that aren't revealed yet
        for y in range(self.game.height):
            for x in range(self.game.width):
                if (self.game.revealed[y][x] and
                    self.game.board[y][x] == 0 and
                    not self.game.game_over):

                    # Find all adjacent hidden cells
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.game.width and
                                0 <= ny < self.game.height and
                                not self.game.revealed[ny][nx] and
                                not self.game.flagged[ny][nx]):
                                return (nx, ny)

        return None

    def _advanced_strategy(self) -> Optional[Tuple[int, int]]:
        """Apply advanced constraint satisfaction strategies."""
        # Group cells by their constraints
        constraints = self._analyze_constraints()

        # Look for cells that must be safe or mines based on overlapping constraints
        for constraint in constraints:
            safe_cells, mine_cells = self._solve_constraint_group(constraint)
            if safe_cells:
                return random.choice(safe_cells)
            if mine_cells:
                mx, my = random.choice(mine_cells)
                self.game.flag(mx, my)
                self.moves_made += 1
                return self.solve()

        return None

    def _analyze_constraints(self) -> List:
        """Analyze the current board state and group related constraints."""
        constraints = []

        # Find all revealed cells with numbers
        numbered_cells = []
        for y in range(self.game.height):
            for x in range(self.game.width):
                if (self.game.revealed[y][x] and
                    self.game.board[y][x] > 0):
                    numbered_cells.append((x, y))

        # Group cells that share hidden neighbors
        visited = set()
        for cell in numbered_cells:
            if cell not in visited:
                constraint_group = self._find_related_cells(cell, numbered_cells)
                if len(constraint_group) > 1:
                    constraints.append(constraint_group)
                visited.update(constraint_group)

        return constraints

    def _find_related_cells(self, start_cell: Tuple[int, int],
                           all_numbered: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Find all numbered cells that share hidden neighbors with the start cell."""
        related = {start_cell}
        frontier = [start_cell]

        while frontier:
            current = frontier.pop()
            current_hidden = set(self._get_hidden_neighbors(*current))

            for other in all_numbered:
                if other not in related:
                    other_hidden = set(self._get_hidden_neighbors(*other))
                    if current_hidden & other_hidden:  # They share hidden neighbors
                        related.add(other)
                        frontier.append(other)

        return related

    def _solve_constraint_group(self, constraint_group: Set[Tuple[int, int]]) -> Tuple[Optional[List], Optional[List]]:
        """Solve a group of related constraints. Returns (safe_cells, mine_cells)."""
        # This is a simplified constraint solver
        # In a full implementation, you'd use more sophisticated algorithms

        hidden_cells = set()
        for cell in constraint_group:
            hidden_cells.update(self._get_hidden_neighbors(*cell))

        # Simple case: if we have isolated constraints
        if len(constraint_group) == 1:
            cell = list(constraint_group)[0]
            flagged_neighbors = self._count_flagged_neighbors(*cell)
            hidden_count = self._count_hidden_neighbors(*cell)
            hidden_neighbors = self._get_hidden_neighbors(*cell)

            if flagged_neighbors == self.game.board[cell[1]][cell[0]]:
                return (hidden_neighbors, None)
            elif flagged_neighbors + hidden_count == self.game.board[cell[1]][cell[0]]:
                return (None, hidden_neighbors)

        return (None, None)

    def _educated_guess(self) -> Optional[Tuple[int, int]]:
        """Make an educated guess when no safe moves are obvious."""
        # Find all hidden, unflagged cells
        candidates = []
        for y in range(self.game.height):
            for x in range(self.game.width):
                if (not self.game.revealed[y][x] and
                    not self.game.flagged[y][x]):
                    candidates.append((x, y))

        if not candidates:
            return None

        # Score each candidate based on how "safe" it looks
        scored_candidates = []
        for x, y in candidates:
            score = self._calculate_safety_score(x, y)
            scored_candidates.append((score, (x, y)))

        # Sort by safety score (higher is safer)
        scored_candidates.sort(reverse=True)

        # Return the safest looking move
        return scored_candidates[0][1]

    def _calculate_safety_score(self, x: int, y: int) -> float:
        """Calculate how safe a cell looks (higher score = safer)."""
        score = 0.0

        # Check adjacent revealed cells
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.game.width and
                    0 <= ny < self.game.height and
                    self.game.revealed[ny][nx]):

                    if self.game.board[ny][nx] == 0:
                        score += 1.0  # Adjacent to empty cell (good)
                    else:
                        # Adjacent to numbered cell
                        flagged_neighbors = self._count_flagged_neighbors(nx, ny)
                        hidden_count = self._count_hidden_neighbors(nx, ny)

                        if flagged_neighbors == self.game.board[ny][nx]:
                            score += 2.0  # All mines found in this area
                        elif hidden_count > 0:
                            # Some mines still hidden
                            remaining_mines = self.game.board[ny][nx] - flagged_neighbors
                            mine_probability = remaining_mines / hidden_count
                            score += 1.0 - mine_probability

        # Prefer cells in the corner or edge (often safer)
        if x == 0 or x == self.game.width - 1 or y == 0 or y == self.game.height - 1:
            score += 0.5

        # Prefer cells not adjacent to flagged mines
        flagged_count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.game.width and
                    0 <= ny < self.game.height and
                    self.game.flagged[ny][nx]):
                    flagged_count += 1

        score -= flagged_count * 0.3  # Penalty for each adjacent flag

        return score

    def _count_flagged_neighbors(self, x: int, y: int) -> int:
        """Count flagged neighbors of a cell."""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.game.width and
                    0 <= ny < self.game.height and
                    self.game.flagged[ny][nx]):
                    count += 1
        return count

    def _count_hidden_neighbors(self, x: int, y: int) -> int:
        """Count the number of hidden neighbors of a cell."""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.game.width and
                    0 <= ny < self.game.height and
                    not self.game.revealed[ny][nx] and
                    not self.game.flagged[ny][nx]):
                    count += 1
        return count

    def _get_hidden_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get list of hidden neighbors of a cell."""
        hidden = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.game.width and
                    0 <= ny < self.game.height and
                    not self.game.revealed[ny][nx] and
                    not self.game.flagged[ny][nx]):
                    hidden.append((nx, ny))
        return hidden

    def play_game(self, max_moves: int = 1000, delay: float = 0.1) -> bool:
        """Play the game automatically. Returns True if won."""
        print("Starting automated Minesweeper solver...")
        print(f"Board size: {self.game.width}x{self.game.height}, Mines: {self.game.num_mines}")
        print()

        move_count = 0
        while not self.game.game_over and move_count < max_moves:
            move = self.solve()

            if move is None:
                print("No safe moves found. Game may be lost.")
                break

            x, y = move
            print(f"Move {move_count + 1}: Revealing ({x}, {y})")

            mine_hit = self.game.reveal(x, y)
            move_count += 1
            self.moves_made = move_count

            if mine_hit:
                print("Hit a mine! Game over.")
                break

            # Print current state
            self.game.print_board()
            print()

            if delay > 0:
                time.sleep(delay)

        if self.game.won:
            print(f"Victory! Solved in {move_count} moves.")
            return True
        else:
            print(f"Game over after {move_count} moves.")
            return False
