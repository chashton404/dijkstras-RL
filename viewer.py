"""
viewer.py
Loads the trained PPO agent and lets you watch it play the game in raylib.

Run after training:
    python viewer.py

Controls:
    SPACE  — step the agent manually (when auto-play is off)
    A      — toggle auto-play
    R      — new random graph
    Q      — quit
"""

import math
import time
import numpy as np
import pyray as rl
from sb3_contrib import MaskablePPO

from game_core import (
    gen_graph, pick_start_end, dijkstra_path, dijkstra_dist,
    path_cost, are_adjacent, adj_matrix, N_NODES, LETTERS,
)
from train import DijkstraGameEnv   # reuse the observation builder

# ── Window ────────────────────────────────────────────────────────────────────
SW, SH    = 1200, 800
MODEL_PATH = "agent"              # matches train.py output
AUTO_DELAY = 0.6                  # seconds between agent steps in auto mode

# ── Palette ───────────────────────────────────────────────────────────────────
BG           = rl.Color(10,  12,  20,  255)
COL_NODE     = rl.Color(30,  32,  50,  255)
COL_BORDER   = rl.Color(70,  75, 110,  255)
COL_EDGE     = rl.Color(55,  58,  90,  255)
COL_START    = rl.Color(50,  220, 140,  255)
COL_END      = rl.Color(240,  90,  90,  255)
COL_AGENT    = rl.Color(180, 120, 255,  255)   # purple — agent's path
COL_PATH_OPT = rl.Color(255, 210,  60,  255)   # gold   — optimal path
COL_FRONTIER = rl.Color(255, 190,  50,  255)
WHITE        = rl.Color(240, 245, 255,  255)
GRAY         = rl.Color(100, 105, 140,  255)
PANEL_BG     = rl.Color(15,  17,  30,  255)
PANEL_LINE   = rl.Color(40,  44,  75,  255)
COL_ACCENT   = rl.Color(94,  234, 212,  255)

def c(col, alpha):
    return rl.Color(col.r, col.g, col.b, max(0, min(255, int(alpha * 255))))

def lerp_c(a, b, t):
    t = max(0.0, min(1.0, t))
    return rl.Color(int(a.r+(b.r-a.r)*t), int(a.g+(b.g-a.g)*t),
                    int(a.b+(b.b-a.b)*t), 255)

# ── Live layout ───────────────────────────────────────────────────────────────
SW = SH = MAP_H = PANEL_Y = PANEL_H = NODE_R = 0

def update_layout():
    global SW, SH, MAP_H, PANEL_Y, PANEL_H, NODE_R
    SW      = rl.get_screen_width()
    SH      = rl.get_screen_height()
    MAP_H   = int(SH * 0.78)
    PANEL_Y = MAP_H
    PANEL_H = SH - MAP_H
    NODE_R  = max(16, int(min(SW, SH) * 0.026))

def node_px(norm_nodes, i):
    nx, ny = norm_nodes[i]
    return int(nx * SW), int(ny * MAP_H)

def ctext(text, cx, y, size, color):
    tw = rl.measure_text(text, size)
    rl.draw_text(text, cx - tw // 2, y, size, color)

# ── Game state ────────────────────────────────────────────────────────────────

class ViewerGame:
    def __init__(self, model):
        self.model    = model
        self.env      = DijkstraGameEnv(N_NODES)
        self.obs      = None
        self.done     = False
        self.auto     = True
        self.auto_t   = 0.0
        self.t        = 0.0        # global anim timer

        # per-episode data
        self.norm_nodes = []
        self.edges      = []
        self.start      = 0
        self.end        = 0
        self.agent_path = []
        self.opt_path   = []
        self.opt_cost   = 0.0
        self.agent_cost = 0.0
        self.last_action = -1
        self.result_msg  = ""
        self.result_col  = WHITE
        self.step_count  = 0

        self.new_game()

    def new_game(self):
        self.obs, _ = self.env.reset()
        # mirror the env's graph into viewer fields
        self.norm_nodes = self.env.norm_nodes
        self.edges      = self.env.edges
        self.start      = self.env.start
        self.end        = self.env.end
        n               = len(self.norm_nodes)
        self.agent_path = [self.start]
        self.opt_path, self.opt_cost = dijkstra_path(
            self.edges, self.start, self.end, n)
        self.agent_cost  = 0.0
        self.done        = False
        self.last_action = -1
        self.result_msg  = ""
        self.auto_t      = 0.0
        self.step_count  = 0
        print(f"\n[VIEWER] New graph | Start: {LETTERS[self.start]}  "
              f"End: {LETTERS[self.end]}  "
              f"Optimal cost: {self.opt_cost}")

    def agent_step(self):
        if self.done:
            return
        action, _ = self.model.predict(
            self.obs,
            deterministic=True,
            action_masks=self.env.action_masks(),
        )
        action = int(action)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated
        self.last_action = action

        if action != self.agent_path[-1]:
            if "invalid" not in info:
                self.agent_path.append(action)

        self.step_count += 1

        if self.done and "user_cost" in info:
            self.agent_cost = info["user_cost"]
            if info["matched"]:
                self.result_msg = f"Agent matched optimal!  Cost {self.agent_cost}"
                self.result_col = COL_START
            else:
                diff = round(self.agent_cost - self.opt_cost, 1)
                self.result_msg = (f"Agent cost {self.agent_cost}  |  "
                                   f"Optimal {self.opt_cost}  |  "
                                   f"+{diff} over optimal")
                self.result_col = COL_FRONTIER
            print(f"[VIEWER] {self.result_msg}")

    def update(self, dt):
        self.t += dt
        if self.auto and not self.done:
            self.auto_t += dt
            if self.auto_t >= AUTO_DELAY:
                self.auto_t -= AUTO_DELAY
                self.agent_step()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, mx, my):
        n = len(self.norm_nodes)

        # dot grid
        for gx in range(0, SW, 40):
            for gy in range(0, MAP_H, 40):
                rl.draw_pixel(gx, gy, c(COL_BORDER, 0.18))

        agent_edges = set()
        for k in range(len(self.agent_path) - 1):
            u, v = self.agent_path[k], self.agent_path[k + 1]
            agent_edges.add((min(u, v), max(u, v)))

        opt_edges = set()
        for k in range(len(self.opt_path) - 1):
            u, v = self.opt_path[k], self.opt_path[k + 1]
            opt_edges.add((min(u, v), max(u, v)))

        # ── Edges ──
        drawn = set()
        for i in range(n):
            xi, yi = node_px(self.norm_nodes, i)
            for j, w in self.edges[i]:
                ek = (min(i, j), max(i, j))
                if ek in drawn: continue
                drawn.add(ek)
                xj, yj = node_px(self.norm_nodes, j)

                if ek in agent_edges and ek in opt_edges:
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 4,
                                    c(COL_PATH_OPT, 0.9))
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 2,
                                    c(COL_AGENT, 0.9))
                elif ek in agent_edges:
                    pulse = 0.7 + 0.3 * math.sin(self.t * 4)
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 3,
                                    c(COL_AGENT, pulse))
                elif ek in opt_edges and self.done:
                    flash = 0.6 + 0.4 * math.sin(self.t * 3)
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 3,
                                    c(COL_PATH_OPT, flash))
                else:
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 2,
                                    COL_EDGE)

                # weight label
                mid_x = (xi + xj) // 2
                mid_y = (yi + yj) // 2
                lbl   = str(round(w, 1))
                lw    = rl.measure_text(lbl, 11)
                rl.draw_rectangle(mid_x-lw//2-3, mid_y-8, lw+6, 16, c(BG, 0.88))
                tcol = (COL_AGENT if ek in agent_edges else
                        COL_PATH_OPT if (ek in opt_edges and self.done)
                        else GRAY)
                rl.draw_text(lbl, mid_x-lw//2, mid_y-6, 11, tcol)

        # ── Nodes ──
        current = self.agent_path[-1] if self.agent_path else self.start
        for i in range(n):
            x, y = node_px(self.norm_nodes, i)
            lbl  = LETTERS[i]

            if i == self.start:
                fill = COL_START; border = c(COL_START, 0.7); ring = COL_START
            elif i == self.end:
                fill = COL_END;   border = c(COL_END,   0.7); ring = COL_END
            elif i == current and not self.done:
                pulse = 0.5 + 0.5 * math.sin(self.t * 5)
                fill  = lerp_c(COL_NODE, COL_AGENT, 0.7)
                border = c(COL_AGENT, pulse); ring = COL_AGENT
            elif i in self.agent_path:
                fill   = lerp_c(COL_NODE, COL_AGENT, 0.4)
                border = c(COL_AGENT, 0.7); ring = None
            else:
                fill   = COL_NODE
                border = COL_BORDER; ring = None

            if ring:
                rl.draw_circle(x, y, NODE_R+8, c(ring, 0.18))
                rl.draw_circle(x, y, NODE_R+4, c(ring, 0.32))
            rl.draw_circle(x, y, NODE_R, fill)
            rl.draw_circle_lines(x, y, NODE_R, border)
            fs = max(12, NODE_R - 6)
            lw = rl.measure_text(lbl, fs)
            rl.draw_text(lbl, x-lw//2, y-fs//2, fs, WHITE)

        # ── Result overlay ──
        if self.done and self.result_msg:
            flash = 0.7 + 0.3 * math.sin(self.t * 2)
            cw    = min(SW - 60, 800); ch = 60
            cx_   = SW//2 - cw//2;    cy_ = MAP_H//2 - ch//2
            rl.draw_rectangle(cx_, cy_, cw, ch, c(PANEL_BG, 0.92))
            rl.draw_rectangle_lines_ex(
                rl.Rectangle(cx_, cy_, cw, ch), 2,
                c(self.result_col, flash))
            ctext(self.result_msg, SW//2, cy_+18, 18, self.result_col)

        # ── Panel ──
        rl.draw_rectangle(0, PANEL_Y, SW, PANEL_H, PANEL_BG)
        rl.draw_line(0, PANEL_Y, SW, PANEL_Y, PANEL_LINE)

        # Agent path chips
        chip_w = 46; chip_h = 34; chip_y = PANEL_Y + 16
        chip_x = 16; arrow_w = 18
        for idx, nd in enumerate(self.agent_path):
            cx_ = chip_x + idx * (chip_w + arrow_w + 6)
            if idx > 0:
                rl.draw_text("->", cx_ - arrow_w, chip_y + chip_h//2 - 8,
                             14, c(GRAY, 0.6))
            lbl  = LETTERS[nd]
            col_ = (COL_START if nd == self.start else
                    COL_END   if nd == self.end   else COL_AGENT)
            rl.draw_rectangle(cx_, chip_y, chip_w, chip_h, c(col_, 0.18))
            rl.draw_rectangle_lines_ex(
                rl.Rectangle(cx_, chip_y, chip_w, chip_h), 1, col_)
            lw = rl.measure_text(lbl, 16)
            rl.draw_text(lbl, cx_ + chip_w//2 - lw//2,
                         chip_y + chip_h//2 - 8, 16, col_)

        # Stats (right side of panel)
        right_x = SW - 320
        rl.draw_text(f"Agent cost : {self.agent_cost if self.done else '...'  }",
                     right_x, PANEL_Y + 12, 14, COL_AGENT)
        rl.draw_text(f"Optimal    : {self.opt_cost}",
                     right_x, PANEL_Y + 32, 14, COL_PATH_OPT)
        rl.draw_text(f"Steps taken: {self.step_count}",
                     right_x, PANEL_Y + 52, 14, c(GRAY, 0.8))

        # Controls hint
        auto_label = "[A] Auto: ON " if self.auto else "[A] Auto: OFF"
        hints = f"{auto_label}   [SPACE] Step   [R] New graph   [Q] Quit"
        hw = rl.measure_text(hints, 13)
        rl.draw_text(hints, SW//2 - hw//2, PANEL_Y + PANEL_H - 22, 13,
                     c(GRAY, 0.7))

        # Title bar
        rl.draw_rectangle(0, 0, SW, 36, c(PANEL_BG, 0.95))
        rl.draw_line(0, 36, SW, 36, PANEL_LINE)
        ctext("Agent Viewer — Are you smarter than Dijkstra?",
              SW//2, 9, 17, c(WHITE, 0.65))

        # Legend
        legend = [(COL_AGENT, "Agent path"), (COL_PATH_OPT, "Optimal path"),
                  (COL_START, "Start"), (COL_END, "End")]
        lx = SW - 30
        for lc, lt in reversed(legend):
            lw = rl.measure_text(lt, 12)
            rl.draw_text(lt, lx - lw, 10, 12, lc)
            rl.draw_circle(lx - lw - 10, 16, 5, lc)
            lx -= lw + 28


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading model from agent.zip ...")
    try:
        model = MaskablePPO.load(MODEL_PATH)
    except FileNotFoundError:
        print("ERROR: agent.zip not found. Run  python train.py  first.")
        return

    print("Model loaded. Launching viewer...")

    rl.init_window(1200, 800, "Dijkstra Agent Viewer")
    rl.set_target_fps(60)
    rl.set_window_state(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    update_layout()

    game = ViewerGame(model)

    while not rl.window_should_close():
        update_layout()
        dt  = rl.get_frame_time()
        mx  = rl.get_mouse_x()
        my  = rl.get_mouse_y()

        # ── Input ──
        if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
            game.new_game()

        if rl.is_key_pressed(rl.KeyboardKey.KEY_A):
            game.auto = not game.auto
            print(f"[VIEWER] Auto-play {'ON' if game.auto else 'OFF'}")

        if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE):
            game.agent_step()

        if rl.is_key_pressed(rl.KeyboardKey.KEY_Q):
            break

        # Auto-advance to next game 2s after finishing
        if game.done:
            game.auto_t += dt
            if game.auto and game.auto_t > 2.0:
                game.new_game()

        game.update(dt)

        rl.begin_drawing()
        rl.clear_background(BG)
        game.draw(mx, my)
        rl.end_drawing()

    rl.close_window()


if __name__ == "__main__":
    main()
