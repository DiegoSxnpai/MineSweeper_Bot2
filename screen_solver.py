"""
Screen-Based Minesweeper Solver
Finds and plays real Minesweeper games on your screen.
"""

import time
import math
from typing import List, Tuple, Optional, Dict, Any

# Required imports for screen automation
import cv2
import numpy as np
import pyautogui
import easyocr
from PIL import Image


class ScreenMinesweeperSolver:
    """Finds and solves real Minesweeper games on screen."""

    def __init__(self):
        print("Initializing OCR...")
        self.reader = easyocr.Reader(['en'])
        self.game_bounds = None
        self.cell_size = 20
        self.grid_offset = (0, 0)

    def find_minesweeper_window(self) -> bool:
        """Find a Minesweeper game on screen."""
        print("Scanning screen for Minesweeper...")

        try:
            # Try Windows-specific window finding
            if self._find_windows_minesweeper():
                print(f"Found Minesweeper window at {self.game_bounds}")
                return True

            # Fallback to screen content detection
            if self._detect_by_screen_content():
                print(f"Found Minesweeper by screen analysis at {self.game_bounds}")
                return True

        except Exception as e:
            print(f"Error finding Minesweeper: {e}")

        print("No Minesweeper game found on screen")
        return False

    def _find_windows_minesweeper(self) -> bool:
        """Find Minesweeper using Windows API."""
        try:
            import win32gui
            import win32process

            def find_minesweeper(hwnd, extra):
                if win32gui.IsWindowVisible(hwnd):
                    window_text = win32gui.GetWindowText(hwnd).lower()
                    if "mine" in window_text or "sweeper" in window_text:
                        extra.append(hwnd)

            windows = []
            win32gui.EnumWindows(find_minesweeper, windows)

            if windows:
                hwnd = windows[0]
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                self.game_bounds = (left, top, right - left, bottom - top)
                return True

        except ImportError:
            pass  # Win32 API not available

        return False

    def _detect_by_screen_content(self) -> bool:
        """Detect Minesweeper by analyzing screen content."""
        # Take screenshot
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

        # Use adaptive thresholding for better grid detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        edges = cv2.Canny(thresh, 50, 150)

        # Find contours that could be game grids
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for rectangular areas that could be game grids
        max_area = 0
        best_rect = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10000:  # Large enough to be a game area
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                if len(approx) == 4:  # Rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    # Look for square-ish rectangles (typical for Minesweeper)
                    if 0.8 < aspect_ratio < 1.3:
                        # Additional check: look for grid patterns inside
                        if self._has_grid_pattern(gray[y:y+h, x:x+w]):
                            if area > max_area:
                                max_area = area
                                best_rect = (x, y, w, h)

        if best_rect:
            x, y, w, h = best_rect
            self.game_bounds = (x, y, w, h)
            print(f"Found game area by content analysis: {x}, {y}, {w}, {h}")
            return True

        return False

    def _has_grid_pattern(self, region: np.ndarray) -> bool:
        """Check if a region contains a grid pattern."""
        # Look for evenly spaced lines
        edges = cv2.Canny(region, 50, 150)

        # Use Hough transform with stricter parameters
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                               minLineLength=30, maxLineGap=5)

        if lines is None or len(lines) < 6:
            return False

        # Count vertical and horizontal lines
        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if length > 40:  # Long enough to be a grid line
                if abs(x1 - x2) < abs(y1 - y2):  # Horizontal line
                    horizontal_lines.append((y1 + y2) // 2)
                else:  # Vertical line
                    vertical_lines.append((x1 + x2) // 2)

        # Check if we have enough evenly spaced lines
        vertical_lines = sorted(list(set(vertical_lines)))
        horizontal_lines = sorted(list(set(horizontal_lines)))

        # Need at least 8 vertical and 8 horizontal lines for a reasonable grid
        return len(vertical_lines) >= 8 and len(horizontal_lines) >= 8

    def capture_game_board(self) -> Optional[np.ndarray]:
        """Capture the current Minesweeper game board."""
        if not self.game_bounds:
            return None

        x, y, w, h = self.game_bounds
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def detect_grid_layout(self, image: np.ndarray) -> Tuple[int, int]:
        """Detect grid dimensions from the game image."""
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

        # Use adaptive thresholding for better grid detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        edges = cv2.Canny(thresh, 50, 150)

        # Find lines with more restrictive parameters
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150,
                               minLineLength=100, maxLineGap=5)

        if lines is None or len(lines) < 4:
            print("Not enough lines detected, trying fallback method...")
            return self._detect_grid_fallback(image)

        # Separate vertical and horizontal lines with better filtering
        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line properties
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 50:  # Too short to be a grid line
                continue

            # Determine if horizontal or vertical
            if abs(x1 - x2) < abs(y1 - y2):  # More vertical change = horizontal line
                if abs(y1 - y2) > 30:  # Must have significant vertical span
                    horizontal_lines.append((y1 + y2) // 2)
            else:  # More horizontal change = vertical line
                if abs(x1 - x2) > 30:  # Must have significant horizontal span
                    vertical_lines.append((x1 + x2) // 2)

        # Remove duplicates and sort
        vertical_lines = sorted(list(set(vertical_lines)))
        horizontal_lines = sorted(list(set(horizontal_lines)))

        print(f"Detected {len(vertical_lines)} vertical lines and {len(horizontal_lines)} horizontal lines")

        if len(vertical_lines) < 3 or len(horizontal_lines) < 3:
            print("Not enough grid lines, trying fallback...")
            return self._detect_grid_fallback(image)

        # Filter lines to find evenly spaced grid lines
        vertical_lines = self._filter_grid_lines(vertical_lines, min_spacing=15, max_spacing=50)
        horizontal_lines = self._filter_grid_lines(horizontal_lines, min_spacing=15, max_spacing=50)

        if len(vertical_lines) < 3 or len(horizontal_lines) < 3:
            return self._detect_grid_fallback(image)

        # Calculate cell size from the most common spacing
        v_spacings = [vertical_lines[i+1] - vertical_lines[i] for i in range(len(vertical_lines)-1)]
        h_spacings = [horizontal_lines[i+1] - horizontal_lines[i] for i in range(len(horizontal_lines)-1)]

        self.cell_size = min(
            np.median(v_spacings) if v_spacings else 20,
            np.median(h_spacings) if h_spacings else 20
        )

        # Set grid offset to the first grid line position
        self.grid_offset = (vertical_lines[0], horizontal_lines[0])

        width = len(vertical_lines) - 1
        height = len(horizontal_lines) - 1

        print(f"Final grid: {width}x{height}, cell size: {int(self.cell_size)}px")
        print(f"Grid offset: {self.grid_offset}")

        return width, height

    def _filter_grid_lines(self, lines: List[int], min_spacing: int = 15, max_spacing: int = 50) -> List[int]:
        """Filter lines to find evenly spaced grid lines."""
        if len(lines) < 3:
            return lines

        # Calculate spacings between consecutive lines
        spacings = [lines[i+1] - lines[i] for i in range(len(lines)-1)]

        # Find the most common spacing (grid cell size)
        spacing_counts = {}
        for spacing in spacings:
            if min_spacing <= spacing <= max_spacing:
                spacing_counts[spacing] = spacing_counts.get(spacing, 0) + 1

        if not spacing_counts:
            return lines

        # Use the most common spacing
        grid_spacing = max(spacing_counts.items(), key=lambda x: x[1])[0]

        # Filter lines to keep only those that are part of the regular grid
        filtered_lines = [lines[0]]  # Always keep the first line

        for i in range(1, len(lines)):
            current_spacing = lines[i] - lines[i-1]
            if abs(current_spacing - grid_spacing) <= 5:  # Allow small tolerance
                filtered_lines.append(lines[i])
            else:
                # Check if this might be a double line or noise
                break

        return filtered_lines

    def _detect_grid_fallback(self, image: np.ndarray) -> Tuple[int, int]:
        """Fallback grid detection using contour analysis."""
        print("Using fallback grid detection...")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

        # Use adaptive thresholding for better grid detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        edges = cv2.Canny(thresh, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for the largest rectangular contour that could be the game grid
        max_area = 0
        best_rect = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Large enough to be a game grid
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                if len(approx) == 4:  # Rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    # Look for square-ish rectangles
                    if 0.7 < aspect_ratio < 1.4:
                        # Check if this region has grid patterns
                        if self._has_grid_pattern(gray[y:y+h, x:x+w]):
                            if area > max_area:
                                max_area = area
                                best_rect = (x, y, w, h)

        if best_rect:
            x, y, w, h = best_rect
            print(f"Found grid contour: {x}, {y}, {w}, {h}")

            # Try to detect actual cell size by analyzing line patterns in this region
            region = gray[y:y+h, x:x+w]
            cell_size = self._detect_cell_size(region)

            if cell_size > 0:
                cols = w // cell_size
                rows = h // cell_size

                # Validate that this gives reasonable grid dimensions
                if 8 <= cols <= 30 and 8 <= rows <= 24:
                    self.cell_size = cell_size
                    self.grid_offset = (x, y)

                    print(f"Detected grid: {cols}x{rows}, cell size: {cell_size}px")
                    return cols, rows

        # Last resort: assume standard Minesweeper sizes
        print("Using default 16x16 grid")
        self.cell_size = 20
        self.grid_offset = (0, 0)
        return 16, 16

    def _detect_cell_size(self, region: np.ndarray) -> int:
        """Detect cell size from grid pattern analysis."""
        # Use Hough transform to find grid lines
        edges = cv2.Canny(region, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                               minLineLength=20, maxLineGap=5)

        if lines is None or len(lines) < 4:
            return 20  # Default

        # Find spacings between lines
        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if length > 30:
                if abs(x1 - x2) < abs(y1 - y2):  # Horizontal line
                    horizontal_lines.append((y1 + y2) // 2)
                else:  # Vertical line
                    vertical_lines.append((x1 + x2) // 2)

        # Remove duplicates and sort
        vertical_lines = sorted(list(set(vertical_lines)))
        horizontal_lines = sorted(list(set(horizontal_lines)))

        if len(vertical_lines) >= 3:
            v_spacings = [vertical_lines[i+1] - vertical_lines[i] for i in range(len(vertical_lines)-1)]
            v_median = np.median(v_spacings)
            if 15 <= v_median <= 50:  # Reasonable cell size range
                return int(v_median)

        if len(horizontal_lines) >= 3:
            h_spacings = [horizontal_lines[i+1] - horizontal_lines[i] for i in range(len(horizontal_lines)-1)]
            h_median = np.median(h_spacings)
            if 15 <= h_median <= 50:  # Reasonable cell size range
                return int(h_median)

        return 20  # Default fallback

    def read_cell_content(self, image: np.ndarray, x: int, y: int) -> Dict[str, Any]:
        """Read the content of a single cell."""
        cell_x = x * self.cell_size
        cell_y = y * self.cell_size

        # Extract cell region (add some padding)
        padding = 2
        cell_image = image[
            cell_y + padding:cell_y + self.cell_size - padding,
            cell_x + padding:cell_x + self.cell_size - padding
        ]

        if cell_image.size == 0:
            return {'type': 'unknown', 'number': 0, 'confidence': 0}

        # Analyze cell
        cell_info = self._analyze_cell(cell_image)
        return cell_info

    def _analyze_cell(self, cell_image: np.ndarray) -> Dict[str, Any]:
        """Analyze a single cell to determine its content."""
        # First check if cell is flagged (red color)
        if self._is_flagged(cell_image):
            return {'type': 'flag', 'number': -2, 'confidence': 0.9}

        # Check if cell contains a mine (black/dark with specific pattern)
        if self._is_mine(cell_image):
            return {'type': 'mine', 'number': -1, 'confidence': 0.9}

        # Check if cell is revealed (has number or is empty)
        if self._is_revealed(cell_image):
            number, confidence = self._read_number(cell_image)
            cell_type = 'revealed' if number > 0 or confidence > 0.5 else 'empty'
            return {'type': cell_type, 'number': number, 'confidence': confidence}

        # Hidden cell
        return {'type': 'hidden', 'number': 0, 'confidence': 0.8}

    def _is_revealed(self, cell_image: np.ndarray) -> bool:
        """Check if cell is revealed (contains numbers or is empty)."""
        # Multiple checks for revealed cells
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

        # Calculate multiple metrics
        mean_brightness = np.mean(gray)
        std_dev = np.std(gray)

        # Check brightness range - revealed cells are usually lighter
        brightness_ok = 130 < mean_brightness < 240

        # Check variation - revealed cells have lower variation (uniform background)
        variation_ok = std_dev < 35

        # Check for number patterns using OCR confidence
        number, confidence = self._read_number(cell_image)
        has_number = number > 0 and confidence > 0.3

        # Additional check: look for Minesweeper-specific patterns
        # Revealed cells often have a light gray background
        light_pixels = np.sum(gray > 180)
        light_ratio = light_pixels / (gray.shape[0] * gray.shape[1])

        background_ok = light_ratio > 0.6  # Mostly light background

        # Cell is revealed if it has good background AND either low variation or a detected number
        return brightness_ok and background_ok and (variation_ok or has_number)

    def _read_number(self, cell_image: np.ndarray) -> Tuple[int, float]:
        """Read number from cell using OCR with better preprocessing."""
        try:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

            # Try multiple preprocessing approaches
            preprocessings = [
                lambda img: cv2.convertScaleAbs(img, alpha=1.8, beta=20),  # High contrast
                lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                lambda img: cv2.convertScaleAbs(img, alpha=2.0, beta=0),   # Maximum contrast
                lambda img: img,  # Raw image as fallback
            ]

            best_result = (0, 0.0)

            for preprocessing in preprocessings:
                try:
                    processed = preprocessing(gray)
                    results = self.reader.readtext(processed)

                    if results:
                        for result in results:
                            text = result[1]
                            confidence = result[2]

                            # Clean and validate the text
                            text = text.strip()
                            if len(text) == 1 and text.isdigit():
                                number = int(text)
                                if 1 <= number <= 8:
                                    if confidence > best_result[1]:
                                        best_result = (number, confidence)

                except Exception:
                    continue

            return best_result

        except Exception as e:
            return 0, 0.0

    def get_game_state(self) -> Optional[Dict[str, Any]]:
        """Get current state of the Minesweeper game."""
        if not self.game_bounds:
            return None

        image = self.capture_game_board()
        if image is None:
            return None

        width, height = self.detect_grid_layout(image)

        # Read all cells
        board = []
        revealed_count = 0
        hidden_count = 0
        flag_count = 0

        for y in range(height):
            row = []
            for x in range(width):
                cell_info = self.read_cell_content(image, x, y)

                # Count cell types for debugging
                if cell_info['type'] == 'revealed' or cell_info['type'] == 'empty':
                    revealed_count += 1
                elif cell_info['type'] == 'hidden':
                    hidden_count += 1
                elif cell_info['type'] == 'flag':
                    flag_count += 1

                row.append(cell_info)
            board.append(row)

        print(f"Cell analysis: {revealed_count} revealed, {hidden_count} hidden, {flag_count} flagged")

        # Debug: print some sample cell readings
        if board:
            sample_cells = []
            for y in range(min(5, height)):
                for x in range(min(5, width)):
                    cell = board[y][x]
                    if cell['type'] == 'revealed' and cell['number'] > 0:
                        sample_cells.append(f"({x},{y}):{cell['number']}")

            if sample_cells:
                print(f"Sample revealed cells: {', '.join(sample_cells[:5])}")

        return {
            'width': width,
            'height': height,
            'board': board,
            'image': image
        }

    def click_cell(self, x: int, y: int, right_click: bool = False) -> bool:
        """Click on a cell in the game."""
        if not self.game_bounds or not self.grid_offset:
            return False

        # Calculate screen coordinates
        screen_x = (self.game_bounds[0] + self.grid_offset[0] +
                   x * self.cell_size + self.cell_size // 2)
        screen_y = (self.game_bounds[1] + self.grid_offset[1] +
                   y * self.cell_size + self.cell_size // 2)

        try:
            if right_click:
                pyautogui.rightClick(screen_x, screen_y)
            else:
                pyautogui.click(screen_x, screen_y)
            return True
        except Exception as e:
            print(f"Click error: {e}")
            return False

    def find_and_solve(self, delay: float = 1.0) -> bool:
        """Find a Minesweeper game and solve it."""
        print("Starting Minesweeper solver...")

        if not self.find_minesweeper_window():
            print("No Minesweeper game found!")
            print("Make sure Minesweeper is running and visible on screen.")
            return False

        print("Game found! Analyzing board...")

        # Get initial game state
        game_state = self.get_game_state()
        if not game_state:
            print("Could not analyze game board")
            return False

        width, height = game_state['width'], game_state['height']
        print(f"Board: {width}x{height} cells")

        # Check if game has started (has revealed cells)
        revealed_count = sum(1 for row in game_state['board'] for cell in row if cell['type'] in ['revealed', 'empty'])

        if revealed_count == 0:
            print("Game hasn't started yet. Making first move...")
            # Make a safe first move (typically center or corner)
            first_move = self._make_first_move(width, height)
            if first_move:
                x, y = first_move
                print(f"Making first move: clicking ({x}, {y})")
                if self.click_cell(x, y, right_click=False):
                    time.sleep(delay * 2)  # Wait longer for first move
                    # Update game state
                    game_state = self.get_game_state()

        # Simple solving strategy: find revealed cells with numbers
        # and make safe moves based on basic logic
        moves_made = 0
        max_moves = 1000

        while moves_made < max_moves:
            print(f"Move {moves_made + 1}: Analyzing board...")

            # Update game state for current analysis
            current_state = self.get_game_state()
            if current_state:
                game_state = current_state

            # Look for obvious safe moves (cells next to revealed 0s)
            safe_move = self._find_safe_move(game_state)
            if safe_move:
                x, y = safe_move
                print(f"Safe move: clicking ({x}, {y})")
                if self.click_cell(x, y, right_click=False):
                    moves_made += 1
                    time.sleep(delay)
                    continue

            # Look for obvious mines (cells that complete number constraints)
            mine_move = self._find_mine_move(game_state)
            if mine_move:
                x, y = mine_move
                print(f"Mine detected: flagging ({x}, {y})")
                if self.click_cell(x, y, right_click=True):
                    moves_made += 1
                    time.sleep(delay)
                    continue

            # If no obvious moves, try educated guess
            guess_move = self._make_educated_guess(game_state)
            if guess_move:
                x, y = guess_move
                print(f"Guessing: clicking ({x}, {y})")
                if self.click_cell(x, y, right_click=False):
                    moves_made += 1
                    time.sleep(delay)
                    continue

            # No more moves available
            print("No more moves available")
            break

        print(f"Solving complete! Made {moves_made} moves.")

        # Final check of game state
        final_state = self.get_game_state()
        if final_state:
            revealed_count = sum(1 for row in final_state['board'] for cell in row if cell['type'] in ['revealed', 'empty'])
            total_cells = width * height
            coverage = (revealed_count / total_cells) * 100

            if coverage > 80:  # Most of the board revealed
                print(f"Likely victory! {coverage:.1f}% board coverage")
                return True
            else:
                print(f"Game ended. {coverage:.1f}% board coverage")

        return moves_made > 0

    def _make_first_move(self, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Make a safe first move when the game hasn't started."""
        # Try center first (often safe)
        center_x, center_y = width // 2, height // 2
        if self._is_valid_coordinate(center_x, center_y, width, height):
            return (center_x, center_y)

        # Try corner as fallback
        return (0, 0)

    def _is_valid_coordinate(self, x: int, y: int, width: int, height: int) -> bool:
        """Check if coordinates are within bounds."""
        return 0 <= x < width and 0 <= y < height

    def _find_safe_move(self, game_state: Dict) -> Optional[Tuple[int, int]]:
        """Find an obvious safe move."""
        width, height = game_state['width'], game_state['height']

        for y in range(height):
            for x in range(width):
                cell = game_state['board'][y][x]

                if cell['type'] == 'revealed' and cell['number'] == 0:
                    # Cell with 0: all adjacent hidden cells are safe
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < width and 0 <= ny < height and
                                game_state['board'][ny][nx]['type'] == 'hidden'):
                                return (nx, ny)

        return None

    def _find_mine_move(self, game_state: Dict) -> Optional[Tuple[int, int]]:
        """Find an obvious mine to flag."""
        width, height = game_state['width'], game_state['height']

        for y in range(height):
            for x in range(width):
                cell = game_state['board'][y][x]

                if cell['type'] == 'revealed' and cell['number'] > 0:
                    # Count flagged and hidden neighbors
                    flagged_count = 0
                    hidden_cells = []

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < width and 0 <= ny < height):
                                neighbor = game_state['board'][ny][nx]
                                if neighbor['type'] == 'flag':
                                    flagged_count += 1
                                elif neighbor['type'] == 'hidden':
                                    hidden_cells.append((nx, ny))

                    # If flagged count equals cell number, all hidden neighbors are mines
                    if flagged_count == cell['number'] and hidden_cells:
                        return hidden_cells[0]

        return None

    def _make_educated_guess(self, game_state: Dict) -> Optional[Tuple[int, int]]:
        """Make an educated guess for the next move."""
        width, height = game_state['width'], game_state['height']

        # Collect all hidden cells and score them
        candidates = []
        for y in range(height):
            for x in range(width):
                cell = game_state['board'][y][x]
                if cell['type'] == 'hidden':
                    score = self._calculate_safety_score(game_state, x, y)
                    candidates.append((score, (x, y)))

        if not candidates:
            return None

        # Sort by safety score (higher is safer)
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _calculate_safety_score(self, game_state: Dict, x: int, y: int) -> float:
        """Calculate how safe a cell looks."""
        score = 0.5  # Base score
        width, height = game_state['width'], game_state['height']

        # Check adjacent revealed cells
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < width and 0 <= ny < height):
                    neighbor = game_state['board'][ny][nx]
                    if neighbor['type'] == 'revealed':
                        if neighbor['number'] == 0:
                            score += 0.3  # Adjacent to empty cell (good)
                        else:
                            score += 0.1  # Adjacent to numbered cell

        # Prefer edge/corner cells (often safer)
        if x == 0 or x == width - 1 or y == 0 or y == height - 1:
            score += 0.2

        # Penalty for being near flagged cells
        flagged_neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < width and 0 <= ny < height and
                    game_state['board'][ny][nx]['type'] == 'flag'):
                    flagged_neighbors += 1

        score -= flagged_neighbors * 0.1

        return score
