"""
=============================================================
  DYNAMIC PATHFINDING AGENT  –  Single-file Pygame App
=============================================================

DEPENDENCIES  (install before running):
  pip install pygame

Python version: 3.8+

HOW TO RUN:
  python dynamic_pathfinding.py

CONTROLS:
  Left-click  : place a wall / set Start (S key held) / set Goal (G key held)
  Right-click : remove a wall
  S + click   : move Start node
  G + click   : move Goal node
  Buttons on right panel handle everything else.
=============================================================
"""

import pygame
import heapq
import random
import time
import sys
import math

# ──────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────
ROWS, COLS       = 25, 25          # grid dimensions
CELL_SIZE        = 28              # pixels per cell
PANEL_WIDTH      = 240             # right-side control panel width
MARGIN           = 2               # gap between cells
FPS              = 60

GRID_W = COLS * CELL_SIZE
GRID_H = ROWS * CELL_SIZE
WIN_W  = GRID_W + PANEL_WIDTH
WIN_H  = GRID_H

# Colours
C_BG         = (15,  17,  26)
C_CELL       = (30,  34,  50)
C_WALL       = (220, 220, 230)
C_START      = (50,  220, 100)
C_GOAL       = (220,  60,  60)
C_FRONTIER   = (240, 200,  40)
C_VISITED    = (60,  120, 220)
C_PATH       = (50,  220, 100)
C_AGENT      = (255, 140,   0)
C_PANEL      = (20,  23,  35)
C_BTN        = (45,  50,  72)
C_BTN_HOV    = (70,  80, 115)
C_BTN_ACT    = (80, 160, 255)
C_TEXT       = (210, 215, 240)
C_SUBTEXT    = (120, 130, 170)
C_ACCENT     = (80,  160, 255)
C_WARN       = (255, 100,  60)

# Agent movement delay (ms between steps)
AGENT_STEP_MS    = 80
# Dynamic obstacle spawn probability per step (0.0 – 1.0)
DYN_PROB         = 0.07

# ──────────────────────────────────────────────
#  NODE CLASS
# ──────────────────────────────────────────────
class Node:
    """Represents one cell in the grid."""
    __slots__ = ('row', 'col', 'g', 'h', 'f', 'parent')

    def __init__(self, row: int, col: int):
        self.row    = row
        self.col    = col
        self.g: float = float('inf')
        self.h: float = 0.0
        self.f: float = float('inf')
        self.parent: 'Node | None' = None

    # heapq comparisons – compare by f, then h (tie-break)
    def __lt__(self, other):  return (self.f, self.h) < (other.f, other.h)
    def __eq__(self, other):  return (self.row, self.col) == (other.row, other.col)
    def __hash__(self):       return hash((self.row, self.col))

    def coords(self):
        return (self.row, self.col)

# ──────────────────────────────────────────────
#  HEURISTICS
# ──────────────────────────────────────────────
def heuristic_manhattan(a: tuple, b: tuple) -> float:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def heuristic_euclidean(a: tuple, b: tuple) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def heuristic_chebyshev(a: tuple, b: tuple) -> float:
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

HEURISTICS = {
    'Manhattan' : heuristic_manhattan,
    'Euclidean' : heuristic_euclidean,
    'Chebyshev' : heuristic_chebyshev,
}

# ──────────────────────────────────────────────
#  GRID UTILITIES
# ──────────────────────────────────────────────
def get_neighbors(row: int, col: int, walls: set) -> list:
    """Return valid 4-directional neighbours (no diagonals)."""
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    result = []
    for dr, dc in directions:
        nr, nc = row+dr, col+dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in walls:
            result.append((nr, nc))
    return result

def build_path(node: Node) -> list:
    path = []
    cur = node
    while cur:
        path.append(cur.coords())
        cur = cur.parent
    path.reverse()
    return path

# ──────────────────────────────────────────────
#  SEARCH ALGORITHMS
# ──────────────────────────────────────────────
def a_star(start: tuple, goal: tuple, walls: set, hfunc) -> tuple:
    """
    A* Search  –  f(n) = g(n) + h(n)
    Returns (path, visited_set, frontier_set, nodes_visited, exec_ms)
    path is empty list if no solution.
    """
    t0 = time.perf_counter()

    start_node        = Node(*start)
    start_node.g      = 0
    start_node.h      = hfunc(start, goal)
    start_node.f      = start_node.g + start_node.h

    open_heap  = [start_node]
    open_dict  = {start: start_node}   # coord -> node (for fast lookup)
    closed     = set()
    visited_order = []

    while open_heap:
        current = heapq.heappop(open_heap)
        cc = current.coords()

        if cc in closed:
            continue
        closed.add(cc)
        visited_order.append(cc)

        if cc == goal:
            path = build_path(current)
            ms   = (time.perf_counter() - t0) * 1000
            return path, closed, set(open_dict.keys()), len(closed), round(ms, 2)

        for nc in get_neighbors(*cc, walls):
            if nc in closed:
                continue
            tentative_g = current.g + 1

            if nc not in open_dict or tentative_g < open_dict[nc].g:
                nb        = Node(*nc)
                nb.g      = tentative_g
                nb.h      = hfunc(nc, goal)
                nb.f      = nb.g + nb.h
                nb.parent = current
                open_dict[nc] = nb
                heapq.heappush(open_heap, nb)

    ms = (time.perf_counter() - t0) * 1000
    return [], closed, set(open_dict.keys()), len(closed), round(ms, 2)


def greedy_bfs(start: tuple, goal: tuple, walls: set, hfunc) -> tuple:
    """
    Greedy Best-First Search  –  f(n) = h(n)
    Returns same tuple structure as a_star.
    """
    t0 = time.perf_counter()

    start_node   = Node(*start)
    start_node.h = hfunc(start, goal)
    start_node.f = start_node.h
    start_node.g = 0

    open_heap = [start_node]
    open_set  = {start}
    closed    = set()

    while open_heap:
        current = heapq.heappop(open_heap)
        cc = current.coords()

        if cc in closed:
            continue
        closed.add(cc)

        if cc == goal:
            path = build_path(current)
            ms   = (time.perf_counter() - t0) * 1000
            return path, closed, open_set, len(closed), round(ms, 2)

        for nc in get_neighbors(*cc, walls):
            if nc in closed and nc not in open_set:
                continue
            if nc not in closed:
                nb        = Node(*nc)
                nb.h      = hfunc(nc, goal)
                nb.f      = nb.h
                nb.g      = current.g + 1
                nb.parent = current
                open_set.add(nc)
                heapq.heappush(open_heap, nb)

    ms = (time.perf_counter() - t0) * 1000
    return [], closed, open_set, len(closed), round(ms, 2)


ALGORITHMS = {
    'A* Search'  : a_star,
    'Greedy BFS' : greedy_bfs,
}

# ──────────────────────────────────────────────
#  BUTTON HELPER
# ──────────────────────────────────────────────
class Button:
    def __init__(self, x, y, w, h, label, toggle=False, active_color=C_BTN_ACT):
        self.rect         = pygame.Rect(x, y, w, h)
        self.label        = label
        self.toggle       = toggle
        self.active       = False
        self.active_color = active_color
        self._hovered     = False

    def draw(self, surf, font):
        if self.active and self.toggle:
            col = self.active_color
        elif self._hovered:
            col = C_BTN_HOV
        else:
            col = C_BTN
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        pygame.draw.rect(surf, C_ACCENT, self.rect, 1, border_radius=6)
        lbl = font.render(self.label, True, C_TEXT)
        surf.blit(lbl, lbl.get_rect(center=self.rect.center))

    def handle_event(self, event) -> bool:
        """Returns True if clicked."""
        if event.type == pygame.MOUSEMOTION:
            self._hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.toggle:
                    self.active = not self.active
                return True
        return False


# ──────────────────────────────────────────────
#  MAIN APPLICATION
# ──────────────────────────────────────────────
class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Dynamic Pathfinding Agent")
        self.clock  = pygame.time.Clock()

        self.font_sm  = pygame.font.SysFont('Consolas', 13)
        self.font_md  = pygame.font.SysFont('Consolas', 15, bold=True)
        self.font_lg  = pygame.font.SysFont('Consolas', 17, bold=True)
        self.font_ttl = pygame.font.SysFont('Consolas', 18, bold=True)

        # Grid state
        self.walls : set  = set()
        self.start : tuple = (2, 2)
        self.goal  : tuple = (ROWS-3, COLS-3)

        # Visualisation layers
        self.visited_cells  : set  = set()
        self.frontier_cells : set  = set()
        self.path_cells     : list = []

        # Agent
        self.agent_pos    : tuple      = None   # current grid cell
        self.agent_path   : list       = []     # remaining path
        self.agent_moving : bool       = False
        self.last_step_ms : int        = 0

        # Metrics
        self.metric_nodes   = 0
        self.metric_cost    = 0
        self.metric_time_ms = 0.0
        self.status_msg     = "Ready"
        self.status_color   = C_TEXT

        # Selections
        self.algo_idx      = 0   # index into ALGORITHMS list
        self.heur_idx      = 0   # index into HEURISTICS list
        self.algo_names    = list(ALGORITHMS.keys())
        self.heur_names    = list(HEURISTICS.keys())

        # Key modifiers
        self.key_s_held = False
        self.key_g_held = False

        self._build_ui()

    # ── UI BUTTONS ───────────────────────────────
    def _build_ui(self):
        px = GRID_W + 16
        bw = PANEL_WIDTH - 32
        bh = 32
        y  = 55

        self.btn_run    = Button(px, y,      bw, bh, "▶  Run Search")         ; y += 42
        self.btn_move   = Button(px, y,      bw, bh, "🤖 Move Agent")          ; y += 42
        self.btn_rand   = Button(px, y,      bw, bh, "🎲 Randomize Map")       ; y += 42
        self.btn_clear  = Button(px, y,      bw, bh, "🗑  Clear All")           ; y += 42
        self.btn_dyn    = Button(px, y,      bw, bh, "⚡ Dynamic Mode", toggle=True) ; y += 42

        y += 10
        # Algorithm selector buttons
        self.algo_btns = []
        for name in self.algo_names:
            b = Button(px, y, bw, bh, name, toggle=True)
            self.algo_btns.append(b)
            y += 38
        self.algo_btns[self.algo_idx].active = True

        y += 10
        # Heuristic selector buttons
        self.heur_btns = []
        for name in self.heur_names:
            b = Button(px, y, bw, bh, name, toggle=True)
            self.heur_btns.append(b)
            y += 38
        self.heur_btns[self.heur_idx].active = True

        self.all_buttons = [
            self.btn_run, self.btn_move, self.btn_rand, self.btn_clear, self.btn_dyn,
            *self.algo_btns, *self.heur_btns
        ]

    # ── DRAW ─────────────────────────────────────
    def draw(self):
        self.screen.fill(C_BG)
        self._draw_grid()
        self._draw_panel()
        pygame.display.flip()

    def _cell_rect(self, row, col):
        x = col * CELL_SIZE + MARGIN
        y = row * CELL_SIZE + MARGIN
        s = CELL_SIZE - MARGIN * 2
        return pygame.Rect(x, y, s, s)

    def _draw_grid(self):
        # Background grid lines
        for r in range(ROWS):
            for c in range(COLS):
                rect  = self._cell_rect(r, c)
                coord = (r, c)
                if coord in self.walls:
                    col = C_WALL
                elif coord in self.visited_cells:
                    col = C_VISITED
                elif coord in self.frontier_cells:
                    col = C_FRONTIER
                elif coord in self.path_cells:
                    col = C_PATH
                else:
                    col = C_CELL
                pygame.draw.rect(self.screen, col, rect, border_radius=3)

        # Start / Goal overlays
        def _draw_marker(coord, color, letter):
            rect = self._cell_rect(*coord)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            lbl = self.font_md.render(letter, True, C_BG)
            self.screen.blit(lbl, lbl.get_rect(center=rect.center))

        _draw_marker(self.start, C_START, 'S')
        _draw_marker(self.goal,  C_GOAL,  'G')

        # Agent
        if self.agent_pos and self.agent_moving:
            rect = self._cell_rect(*self.agent_pos)
            pygame.draw.rect(self.screen, C_AGENT, rect, border_radius=3)
            lbl = self.font_md.render('A', True, C_BG)
            self.screen.blit(lbl, lbl.get_rect(center=rect.center))

    def _draw_panel(self):
        # Panel background
        panel = pygame.Rect(GRID_W, 0, PANEL_WIDTH, WIN_H)
        pygame.draw.rect(self.screen, C_PANEL, panel)
        pygame.draw.line(self.screen, C_ACCENT, (GRID_W, 0), (GRID_W, WIN_H), 1)

        # Title
        ttl = self.font_ttl.render("PATHFINDER", True, C_ACCENT)
        self.screen.blit(ttl, (GRID_W + 16, 14))

        # Buttons
        for btn in self.all_buttons:
            btn.draw(self.screen, self.font_sm)

        # Section labels
        def _label(text, y):
            lbl = self.font_sm.render(text, True, C_SUBTEXT)
            self.screen.blit(lbl, (GRID_W + 16, y))

        btn_algo_y = self.algo_btns[0].rect.y - 18
        btn_heur_y = self.heur_btns[0].rect.y - 18
        _label("── ALGORITHM ──", btn_algo_y)
        _label("── HEURISTIC ──",  btn_heur_y)

        # Metrics dashboard
        my = WIN_H - 135
        pygame.draw.line(self.screen, C_BTN, (GRID_W+10, my-8), (WIN_W-10, my-8), 1)
        _label("── METRICS ──", my)
        my += 20

        def _metric(label, value, color=C_TEXT):
            k = self.font_sm.render(label, True, C_SUBTEXT)
            v = self.font_md.render(str(value), True, color)
            self.screen.blit(k, (GRID_W+16, my))
            self.screen.blit(v, (GRID_W+16, my+14))
            return 32

        my += _metric("Nodes Visited",  self.metric_nodes)
        my += _metric("Path Cost",       self.metric_cost)
        my += _metric("Time (ms)",        f"{self.metric_time_ms:.2f}")

        # Status message
        smsg = self.font_sm.render(self.status_msg, True, self.status_color)
        self.screen.blit(smsg, smsg.get_rect(centerx=GRID_W + PANEL_WIDTH//2, y=WIN_H-18))

    # ── INTERACTIONS ─────────────────────────────
    def _grid_coord(self, mx, my):
        if mx >= GRID_W:
            return None
        c = mx // CELL_SIZE
        r = my // CELL_SIZE
        if 0 <= r < ROWS and 0 <= c < COLS:
            return (r, c)
        return None

    def _run_search(self, start_override=None):
        """Run selected algorithm; populate visualisation layers & metrics."""
        src = start_override if start_override else self.start
        algo = ALGORITHMS[self.algo_names[self.algo_idx]]
        hfun = HEURISTICS[self.heur_names[self.heur_idx]]

        path, visited, frontier, n_vis, t_ms = algo(src, self.goal, self.walls, hfun)

        self.visited_cells  = visited
        self.frontier_cells = frontier
        self.metric_nodes   = n_vis
        self.metric_time_ms = t_ms

        if path:
            self.path_cells   = set(path)
            self.metric_cost  = len(path) - 1
            self.status_msg   = f"Path found! ({self.algo_names[self.algo_idx]})"
            self.status_color = C_START
        else:
            self.path_cells   = set()
            self.metric_cost  = 0
            self.status_msg   = "No Path Found!"
            self.status_color = C_WARN

        return path

    def _start_agent(self):
        path = self._run_search()
        if path:
            self.agent_pos    = self.start
            self.agent_path   = path[1:]   # exclude start (already there)
            self.agent_moving = True
            self.last_step_ms = pygame.time.get_ticks()
            self.status_msg   = "Agent moving…"
            self.status_color = C_AGENT
        else:
            self.agent_moving = False

    def _step_agent(self):
        """Advance the agent one cell if time has elapsed."""
        now = pygame.time.get_ticks()
        if now - self.last_step_ms < AGENT_STEP_MS:
            return
        self.last_step_ms = now

        # Dynamic mode: spawn random obstacles
        if self.btn_dyn.active:
            spawned = []
            for r in range(ROWS):
                for c in range(COLS):
                    coord = (r, c)
                    if (coord not in self.walls and
                            coord != self.start and
                            coord != self.goal and
                            coord != self.agent_pos and
                            random.random() < DYN_PROB / (ROWS * COLS / 5)):
                        spawned.append(coord)
            # Only add a small number so the grid doesn't fill instantly
            spawn_count = random.randint(0, 2)
            for coord in random.sample(spawned, min(spawn_count, len(spawned))):
                self.walls.add(coord)

            # Check if any newly spawned wall is on current path
            remaining_set = set(self.agent_path)
            if remaining_set & self.walls:
                # Re-plan from current position
                new_path = self._run_search(start_override=self.agent_pos)
                if new_path:
                    self.agent_path = new_path[1:]
                    self.status_msg   = "Re-planned! ↺"
                    self.status_color = C_WARN
                else:
                    self.agent_moving = False
                    self.status_msg   = "Blocked – No Path!"
                    self.status_color = C_WARN
                    return

        if not self.agent_path:
            self.agent_moving = False
            self.agent_pos    = self.goal
            self.status_msg   = "Goal reached! ✓"
            self.status_color = C_START
            return

        next_cell = self.agent_path.pop(0)

        # Safety: next cell might have become a wall
        if next_cell in self.walls:
            new_path = self._run_search(start_override=self.agent_pos)
            if new_path:
                self.agent_path = new_path[1:]
                self.status_msg = "Re-planned! ↺"
                self.status_color = C_WARN
            else:
                self.agent_moving = False
                self.status_msg   = "Blocked – No Path!"
                self.status_color = C_WARN
            return

        self.agent_pos = next_cell

    def _randomize_map(self):
        self.walls.clear()
        self._clear_viz()
        density = 0.30
        for r in range(ROWS):
            for c in range(COLS):
                coord = (r, c)
                if coord == self.start or coord == self.goal:
                    continue
                if random.random() < density:
                    self.walls.add(coord)
        self.status_msg   = "Map randomized"
        self.status_color = C_TEXT

    def _clear_all(self):
        self.walls.clear()
        self._clear_viz()
        self.agent_moving = False
        self.agent_pos    = None
        self.status_msg   = "Cleared"
        self.status_color = C_TEXT

    def _clear_viz(self):
        self.visited_cells  = set()
        self.frontier_cells = set()
        self.path_cells     = set()
        self.metric_nodes   = 0
        self.metric_cost    = 0
        self.metric_time_ms = 0.0

    # ── EVENT HANDLING ───────────────────────────
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s: self.key_s_held = True
                if event.key == pygame.K_g: self.key_g_held = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_s: self.key_s_held = False
                if event.key == pygame.K_g: self.key_g_held = False

            # Grid mouse interaction
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
                mx, my = pygame.mouse.get_pos()
                coord  = self._grid_coord(mx, my)
                if coord:
                    lb, _, rb = pygame.mouse.get_pressed()
                    if lb:
                        if self.key_s_held:
                            if coord not in self.walls and coord != self.goal:
                                self.start = coord; self._clear_viz()
                        elif self.key_g_held:
                            if coord not in self.walls and coord != self.start:
                                self.goal = coord; self._clear_viz()
                        else:
                            if coord != self.start and coord != self.goal:
                                self.walls.add(coord); self._clear_viz()
                    if rb:
                        self.walls.discard(coord)

            # Algorithm selector
            for i, btn in enumerate(self.algo_btns):
                if btn.handle_event(event):
                    for b in self.algo_btns: b.active = False
                    btn.active    = True
                    self.algo_idx = i

            # Heuristic selector
            for i, btn in enumerate(self.heur_btns):
                if btn.handle_event(event):
                    for b in self.heur_btns: b.active = False
                    btn.active    = True
                    self.heur_idx = i

            # Action buttons
            if self.btn_run.handle_event(event):
                self.agent_moving = False
                self._run_search()

            if self.btn_move.handle_event(event):
                if not self.agent_moving:
                    self._start_agent()

            if self.btn_rand.handle_event(event):
                self.agent_moving = False
                self._randomize_map()

            if self.btn_clear.handle_event(event):
                self._clear_all()

            self.btn_dyn.handle_event(event)

    # ── MAIN LOOP ────────────────────────────────
    def run(self):
        while True:
            self.handle_events()

            if self.agent_moving:
                self._step_agent()

            self.draw()
            self.clock.tick(FPS)


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == '__main__':
    App().run()