import pyray as rl
import math
import random
import heapq

random.seed()

# ── Palette ───────────────────────────────────────────────────────────────────
BG           = rl.Color(10,  12,  20,  255)
COL_NODE     = rl.Color(30,  32,  50,  255)
COL_BORDER   = rl.Color(70,  75, 110,  255)
COL_EDGE     = rl.Color(55,  58,  90,  255)
COL_START    = rl.Color(50,  220, 140,  255)
COL_END      = rl.Color(240,  90,  90,  255)
COL_SELECTED = rl.Color(120, 160, 255,  255)
COL_PATH_USR = rl.Color(120, 160, 255,  255)
COL_PATH_OPT = rl.Color(255, 210,  60,  255)
WHITE        = rl.Color(240, 245, 255,  255)
GRAY         = rl.Color(100, 105, 140,  255)
PANEL_BG     = rl.Color(15,  17,  30,  255)
PANEL_LINE   = rl.Color(40,  44,  75,  255)
COL_BTN      = rl.Color(80,  160, 255,  255)
COL_BTN_RED  = rl.Color(220,  80,  80,  255)
COL_ACCENT   = rl.Color(94,  234, 212,  255)
COL_FRONTIER = rl.Color(255, 190,  50,  255)
COL_SETTLED  = rl.Color(40,   60,  90,  255)

def c(color, alpha):
    return rl.Color(color.r, color.g, color.b, max(0, min(255, int(alpha * 255))))

def lerp_c(a, b, t):
    t = max(0.0, min(1.0, t))
    return rl.Color(int(a.r+(b.r-a.r)*t), int(a.g+(b.g-a.g)*t),
                    int(a.b+(b.b-a.b)*t), 255)

# ── Config ────────────────────────────────────────────────────────────────────
N_NODES = 24
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
ANIM_STEP_DELAY = 0.18   # seconds between Dijkstra animation steps

# ── Game states ───────────────────────────────────────────────────────────────
SCR_TITLE    = "title"
SCR_WHAT     = "what"
SCR_GAME     = "game"

GS_COUNTDOWN = "countdown"   # 3..2..1 before user can interact
GS_BUILDING  = "building"    # user is picking nodes, timer running
GS_ANIMATING = "animating"   # Dijkstra animation playing
GS_RESULTS   = "results"     # side-by-side comparison shown

# ── Live layout ───────────────────────────────────────────────────────────────
SW = SH = MAP_H = PANEL_Y = PANEL_H = NODE_R = MARGIN = 0

def update_layout():
    global SW, SH, MAP_H, PANEL_Y, PANEL_H, NODE_R, MARGIN
    SW      = rl.get_screen_width()
    SH      = rl.get_screen_height()
    MAP_H   = int(SH * 0.72)
    PANEL_Y = MAP_H
    PANEL_H = SH - MAP_H
    NODE_R  = max(16, int(min(SW, SH) * 0.026))
    MARGIN  = max(60, int(min(SW, SH) * 0.08))

# ── Graph generation ──────────────────────────────────────────────────────────
def gen_graph():
    REF_W, REF_H = 1200.0, 576.0
    pad, min_sep  = 0.08, 0.13
    nodes, attempts = [], 0
    while len(nodes) < N_NODES and attempts < 8000:
        attempts += 1
        nx = random.uniform(pad, 1.0 - pad)
        ny = random.uniform(pad, 1.0 - pad)
        if all(math.hypot(nx-ex, ny-ey) > min_sep for ex, ey in nodes):
            nodes.append((nx, ny))

    n = len(nodes)
    edges = [[] for _ in range(n)]

    def add_edge(i, j):
        nxi, nyi = nodes[i]; nxj, nyj = nodes[j]
        w = round(math.hypot(nxi*REF_W - nxj*REF_W, nyi*REF_H - nyj*REF_H) / 40.0, 1)
        if not any(nb == j for nb, _ in edges[i]):
            edges[i].append((j, w)); edges[j].append((i, w))

    unvisited = list(range(1, n)); visited = [0]
    while unvisited:
        i = random.choice(visited); j = random.choice(unvisited)
        add_edge(i, j); visited.append(j); unvisited.remove(j)

    for i in range(n):
        for j in range(i+1, n):
            if math.hypot(nodes[i][0]-nodes[j][0], nodes[i][1]-nodes[j][1]) < 0.38:
                if random.random() < 0.32:
                    add_edge(i, j)

    return nodes, edges

def node_px(norm_nodes, i):
    nx, ny = norm_nodes[i]
    return int(nx * SW), int(ny * MAP_H)

# ── Graph algorithms ──────────────────────────────────────────────────────────
def pick_start_end(norm_nodes, edges):
    n = len(norm_nodes)
    adj = set()
    for i in range(n):
        for j, _ in edges[i]: adj.add((min(i,j), max(i,j)))
    for _ in range(500):
        s = random.randint(0, n-1); e = random.randint(0, n-1)
        if s == e: continue
        if (min(s,e), max(s,e)) in adj: continue
        if _dijkstra_dist(edges, s, e, n) < float('inf'): return s, e
    return 0, n-1

def _dijkstra_dist(edges, src, dst, n):
    dist = [float('inf')]*n; dist[src] = 0
    heap = [(0, src)]; settled = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in settled: continue
        settled.add(u)
        if u == dst: return d
        for v, w in edges[u]:
            if v not in settled and d+w < dist[v]:
                dist[v] = d+w; heapq.heappush(heap, (dist[v], v))
    return float('inf')

def dijkstra_path(edges, src, dst, n):
    dist = [float('inf')]*n; prev = [-1]*n; dist[src] = 0
    heap = [(0, src)]; settled = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in settled: continue
        settled.add(u)
        if u == dst: break
        for v, w in edges[u]:
            if v not in settled and d+w < dist[v]:
                dist[v] = d+w; prev[v] = u; heapq.heappush(heap, (dist[v], v))
    path = []; cur = dst
    while cur != -1: path.append(cur); cur = prev[cur]
    path.reverse()
    if path and path[0] == src: return path, round(dist[dst], 1)
    return [], float('inf')

def build_anim_steps(edges, src, dst, n):
    """Build a list of animation events for Dijkstra's algorithm."""
    dist = [float('inf')]*n; prev = [-1]*n; dist[src] = 0
    heap = [(0, src)]; settled = set(); steps = []
    steps.append({"t": "init", "src": src})
    while heap:
        d, u = heapq.heappop(heap)
        if u in settled: continue
        settled.add(u)
        steps.append({"t": "settle", "u": u})
        if u == dst: break
        for v, w in edges[u]:
            if v in settled: continue
            steps.append({"t": "relax", "u": u, "v": v})
            if d+w < dist[v]:
                dist[v] = d+w; prev[v] = u
                heapq.heappush(heap, (dist[v], v))
                steps.append({"t": "update", "v": v})
    path = []; cur = dst
    while cur != -1: path.append(cur); cur = prev[cur]
    path.reverse()
    if path and path[0] == src:
        steps.append({"t": "path", "path": path})
    return steps

def path_cost(edges, path):
    total = 0.0
    for k in range(len(path)-1):
        u, v = path[k], path[k+1]
        found = next((w for nb, w in edges[u] if nb == v), None)
        if found is None: return float('inf')
        total += found
    return round(total, 1)

def are_adjacent(edges, u, v):
    return any(nb == v for nb, _ in edges[u])

# ── Draw helpers ──────────────────────────────────────────────────────────────
def draw_btn(bx, by, bw, bh, label, active, hovered, font_size=20, active_col=None):
    ac = active_col or COL_BTN
    bg = c(ac, 0.18) if (active and hovered) else (c(ac, 0.08) if active else c(PANEL_BG, 0.95))
    bc = ac if (active and hovered) else (c(ac, 0.55) if active else PANEL_LINE)
    tc = WHITE if active else c(GRAY, 0.5)
    rl.draw_rectangle(bx, by, bw, bh, bg)
    rl.draw_rectangle_lines_ex(rl.Rectangle(bx, by, bw, bh), 2, bc)
    lw = rl.measure_text(label, font_size)
    rl.draw_text(label, bx+bw//2-lw//2, by+bh//2-font_size//2, font_size, tc)

def ctext(text, cx, y, size, color):
    tw = rl.measure_text(text, size)
    rl.draw_text(text, cx - tw//2, y, size, color)

def node_at_px(norm_nodes, mx, my):
    for i in range(len(norm_nodes)):
        x, y = node_px(norm_nodes, i)
        if y > MAP_H: continue
        if math.hypot(mx-x, my-y) <= NODE_R+6: return i
    return -1

def draw_node_shape(x, y, label, fill, border, ring_col=None, hov=False):
    if ring_col:
        rl.draw_circle(x, y, NODE_R+8, c(ring_col, 0.18))
        rl.draw_circle(x, y, NODE_R+4, c(ring_col, 0.32))
    if hov:
        rl.draw_circle(x, y, NODE_R+5, c(WHITE, 0.12))
    rl.draw_circle(x, y, NODE_R, fill)
    rl.draw_circle_lines(x, y, NODE_R, border)
    fs = max(12, NODE_R-6)
    lw = rl.measure_text(label, fs)
    rl.draw_text(label, x-lw//2, y-fs//2, fs, WHITE)

def fmt_time(seconds):
    return f"{seconds:.1f}s"

# ── Map drawing ───────────────────────────────────────────────────────────────
def draw_map_edges(norm_nodes, edges, n, user_edges, opt_edges, anim_edge, t,
                   mode="building"):
    """Draw all edges. mode = building | animating | results"""
    drawn = set()
    for i in range(n):
        xi, yi = node_px(norm_nodes, i)
        for j, w in edges[i]:
            ek = (min(i,j), max(i,j))
            if ek in drawn: continue
            drawn.add(ek)
            xj, yj = node_px(norm_nodes, j)

            if mode == "results":
                if ek in opt_edges and ek in user_edges:
                    # both paths share this edge — draw layered
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 4, c(COL_PATH_OPT, 0.9))
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 2, c(COL_PATH_USR, 0.9))
                elif ek in opt_edges:
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 4, c(COL_PATH_OPT, 0.85))
                elif ek in user_edges:
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 3, c(COL_PATH_USR, 0.7))
                else:
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 2, c(COL_EDGE, 0.35))
            elif mode == "animating":
                if ek == anim_edge:
                    pulse = 0.7 + 0.3*math.sin(t*10)
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 3, c(COL_FRONTIER, pulse))
                elif ek in opt_edges:
                    flash = 0.6 + 0.4*math.sin(t*4)
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 4, c(COL_PATH_OPT, flash))
                elif ek in user_edges:
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 3, c(COL_PATH_USR, 0.5))
                else:
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 2, COL_EDGE)
            else:  # building
                if ek in user_edges:
                    pulse = 0.7 + 0.3*math.sin(t*5)
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 3, c(COL_PATH_USR, pulse))
                else:
                    rl.draw_line_ex(rl.Vector2(xi,yi), rl.Vector2(xj,yj), 2, COL_EDGE)

            # weight label
            mid_x = (xi+xj)//2; mid_y = (yi+yj)//2
            lbl = str(round(w, 1))
            lw  = rl.measure_text(lbl, 11)
            rl.draw_rectangle(mid_x-lw//2-3, mid_y-8, lw+6, 16, c(BG, 0.88))
            if mode == "results":
                if ek in opt_edges: tcol = COL_PATH_OPT
                elif ek in user_edges: tcol = COL_PATH_USR
                else: tcol = c(GRAY, 0.4)
            else:
                tcol = COL_PATH_USR if ek in user_edges else GRAY
            rl.draw_text(lbl, mid_x-lw//2, mid_y-6, 11, tcol)


def draw_map_nodes(norm_nodes, n, start_node, end_node, user_path,
                   hover_node, anim_node_status, opt_path_set, t, mode="building"):
    for i in range(n):
        x, y = node_px(norm_nodes, i)
        lbl  = LETTERS[i]

        if mode == "results":
            in_opt  = i in opt_path_set
            in_user = i in user_path
            if i == start_node:
                fill = COL_START; border = c(COL_START, 0.7); ring = COL_START
            elif i == end_node:
                fill = COL_END;   border = c(COL_END,   0.7); ring = COL_END
            elif in_opt and in_user:
                fill = lerp_c(COL_PATH_OPT, COL_PATH_USR, 0.5)
                border = COL_PATH_OPT; ring = None
            elif in_opt:
                fill = lerp_c(COL_NODE, COL_PATH_OPT, 0.6)
                border = COL_PATH_OPT; ring = None
            elif in_user:
                fill = lerp_c(COL_NODE, COL_PATH_USR, 0.5)
                border = COL_PATH_USR; ring = None
            else:
                fill = c(COL_NODE, 0.4); border = c(COL_BORDER, 0.4); ring = None
        elif mode == "animating":
            ns = anim_node_status.get(i, "default")
            if i == start_node:
                fill = COL_START; border = c(COL_START, 0.7); ring = COL_START
            elif i == end_node:
                fill = COL_END;   border = c(COL_END,   0.7); ring = COL_END
            elif ns == "path":
                flash = 0.6+0.4*math.sin(t*4)
                fill  = lerp_c(COL_NODE, COL_PATH_OPT, 0.6)
                border = c(COL_PATH_OPT, flash); ring = None
            elif ns == "frontier":
                pulse = 0.5+0.5*math.sin(t*4+i)
                fill  = lerp_c(COL_NODE, COL_FRONTIER, 0.3)
                border = c(COL_FRONTIER, pulse); ring = None
            elif ns == "settled":
                fill = COL_SETTLED; border = c(COL_BORDER, 0.5); ring = None
            else:
                fill = COL_NODE; border = COL_BORDER; ring = None
        else:  # building
            if i == start_node:
                fill = COL_START; border = c(COL_START, 0.6); ring = COL_START
            elif i == end_node:
                fill = COL_END;   border = c(COL_END,   0.6); ring = COL_END
            elif i in user_path:
                fill = lerp_c(COL_NODE, COL_SELECTED, 0.5)
                border = COL_SELECTED; ring = None
            else:
                fill   = COL_NODE
                border = c(WHITE, 0.6) if hover_node == i else COL_BORDER
                ring   = None

        draw_node_shape(x, y, lbl, fill, border, ring_col=ring,
                        hov=(hover_node == i and mode == "building"))


# ── Panel ─────────────────────────────────────────────────────────────────────
def draw_panel(start_node, end_node, user_path, game_state,
               user_time, mx, my):
    rl.draw_rectangle(0, PANEL_Y, SW, PANEL_H, PANEL_BG)
    rl.draw_line(0, PANEL_Y, SW, PANEL_Y, PANEL_LINE)

    if game_state in (GS_COUNTDOWN, GS_BUILDING):
        rl.draw_text("Node Order", 24, PANEL_Y+14, 14, c(GRAY, 0.8))

        chip_w=54; chip_h=38; chip_y=PANEL_Y+36; chip_x0=24; gap=10; arrow_w=22
        for idx, node_i in enumerate(user_path):
            cx = chip_x0 + idx*(chip_w+gap+arrow_w)
            if idx > 0:
                rl.draw_text("->", cx-arrow_w, chip_y+chip_h//2-8, 16, c(GRAY, 0.7))
            lbl  = LETTERS[node_i]
            fill = (COL_START if node_i == start_node
                    else COL_END if node_i == end_node
                    else c(COL_SELECTED, 0.8))
            rl.draw_rectangle(cx, chip_y, chip_w, chip_h, c(fill, 0.18))
            rl.draw_rectangle_lines_ex(rl.Rectangle(cx, chip_y, chip_w, chip_h), 2, fill)
            lw = rl.measure_text(lbl, 18)
            rl.draw_text(lbl, cx+chip_w//2-lw//2, chip_y+chip_h//2-9, 18, fill)

        if len(user_path) == 1:
            cx = chip_x0+(chip_w+gap+arrow_w)
            rl.draw_text("->", cx-arrow_w, chip_y+chip_h//2-8, 16, c(GRAY, 0.4))
            rl.draw_rectangle_lines_ex(rl.Rectangle(cx, chip_y, chip_w, chip_h), 1, c(GRAY, 0.3))
            rl.draw_text("?", cx+chip_w//2-6, chip_y+chip_h//2-9, 18, c(GRAY, 0.3))

        btn_w=140; btn_h=40; btn_right=SW-24
        btn_y = PANEL_Y+PANEL_H//2-btn_h//2

        sub_bx = btn_right-btn_w; sub_by = btn_y
        can_submit = (len(user_path) >= 2 and user_path[-1] == end_node
                      and game_state == GS_BUILDING)
        sub_hov = sub_bx <= mx <= sub_bx+btn_w and sub_by <= my <= sub_by+btn_h
        draw_btn(sub_bx, sub_by, btn_w, btn_h, "Submit", can_submit, sub_hov, 17, COL_START)

        del_bx = btn_right-btn_w*2-14; del_by = btn_y
        can_del = len(user_path) > 1 and game_state == GS_BUILDING
        del_hov = del_bx <= mx <= del_bx+btn_w and del_by <= my <= del_by+btn_h
        draw_btn(del_bx, del_by, btn_w, btn_h, "<- Delete", can_del, del_hov, 17, COL_BTN_RED)

        if game_state == GS_BUILDING:
            last = user_path[-1] if user_path else -1
            if last == end_node and len(user_path) > 1:
                hint = "Path complete - press Submit!"; hcol = COL_START
            elif last != -1:
                hint = f"Click a node adjacent to {LETTERS[last]} to continue"
                hcol = c(GRAY, 0.8)
            else:
                hint = ""; hcol = WHITE
            hw = rl.measure_text(hint, 13)
            rl.draw_text(hint, SW//2-hw//2, PANEL_Y+PANEL_H-22, 13, hcol)

        return sub_bx, sub_by, btn_w, btn_h, sub_hov, can_submit, \
               del_bx, del_by, del_hov, can_del

    elif game_state == GS_ANIMATING:
        ctext("Dijkstra is finding the optimal path...",
              SW//2, PANEL_Y+PANEL_H//2-10, 16, c(COL_FRONTIER, 0.9))
        return None

    elif game_state == GS_RESULTS:
        ctext("Dijkstra animation complete — see results above.",
              SW//2, PANEL_Y+PANEL_H//2-10, 15, c(GRAY, 0.7))
        return None

    return None


# ── Title background — Dijkstra simulation ────────────────────────────────────
_BG_N = 16   # nodes in the background graph

def _make_bg_graph():
    random.seed(42)
    pts = []
    for _ in range(4000):
        if len(pts) >= _BG_N: break
        nx, ny = random.random(), random.random()
        if all(math.hypot(nx-ex, ny-ey) > 0.19 for ex, ey in pts):
            pts.append((nx, ny))
    random.seed()
    # Build adjacency using a spanning tree + extras so graph is always connected
    adj = [[] for _ in range(len(pts))]
    def bg_add(i, j):
        if not any(nb == j for nb, _ in adj[i]):
            nxi, nyi = pts[i]; nxj, nyj = pts[j]
            w = round(math.hypot(nxi-nxj, nyi-nyj), 3)
            adj[i].append((j, w)); adj[j].append((i, w))
    unvis = list(range(1, len(pts))); vis = [0]
    while unvis:
        i = random.choice(vis); j = random.choice(unvis)
        bg_add(i, j); vis.append(j); unvis.remove(j)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1]) < 0.42:
                if random.random() < 0.45:
                    bg_add(i, j)
    return pts, adj

_BG_PTS, _BG_ADJ = _make_bg_graph()

# ── Dijkstra simulation state ─────────────────────────────────────────────────
# Phases: "running" -> "path_flash" -> "fade_out" -> restart
_bg_phase       = "running"
_bg_step_timer  = 0.0
_bg_step_delay  = 0.22      # seconds per Dijkstra step
_bg_flash_timer = 0.0
_bg_fade_timer  = 0.0
_bg_src         = 0
_bg_dst         = 0
_bg_node_status = {}        # i -> "frontier" | "settled" | "path"
_bg_path_edges  = set()     # edges on final optimal path
_bg_settle_edges= set()     # edges that have been relaxed (settled side)
_bg_steps       = []        # pre-built step list
_bg_step_idx    = 0
_BG_FLASH_DUR   = 1.8       # seconds to show path flash
_BG_FADE_DUR    = 0.8       # seconds to fade everything out before restart


def _bg_dijkstra_steps(adj, src, dst):
    n = len(adj)
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[src] = 0
    heap = [(0, src)]
    settled = set()
    steps = [{"t": "init", "src": src}]
    while heap:
        d, u = heapq.heappop(heap)
        if u in settled: continue
        settled.add(u)
        steps.append({"t": "settle", "u": u})
        if u == dst: break
        for v, w in adj[u]:
            if v in settled: continue
            steps.append({"t": "relax", "u": u, "v": v})
            if d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))
                steps.append({"t": "update", "v": v})
    path = []; cur = dst
    while cur != -1: path.append(cur); cur = prev[cur]
    path.reverse()
    if path and path[0] == src:
        steps.append({"t": "path", "path": path})
    return steps


def _bg_restart():
    global _bg_src, _bg_dst, _bg_steps, _bg_step_idx, _bg_phase
    global _bg_node_status, _bg_path_edges, _bg_settle_edges
    global _bg_step_timer, _bg_flash_timer, _bg_fade_timer
    n = len(_BG_PTS)
    _bg_src = random.randint(0, n-1)
    _bg_dst = random.randint(0, n-1)
    while _bg_dst == _bg_src: _bg_dst = random.randint(0, n-1)
    _bg_steps       = _bg_dijkstra_steps(_BG_ADJ, _bg_src, _bg_dst)
    _bg_step_idx    = 0
    _bg_node_status = {}
    _bg_path_edges  = set()
    _bg_settle_edges= set()
    _bg_phase       = "running"
    _bg_step_timer  = 0.0
    _bg_flash_timer = 0.0
    _bg_fade_timer  = 0.0

# Initialise immediately
_bg_restart()


def _update_bg(dt):
    global _bg_phase, _bg_step_timer, _bg_flash_timer, _bg_fade_timer
    global _bg_step_idx, _bg_node_status, _bg_path_edges, _bg_settle_edges

    if _bg_phase == "running":
        _bg_step_timer += dt
        while _bg_step_timer >= _bg_step_delay and _bg_step_idx < len(_bg_steps):
            step = _bg_steps[_bg_step_idx]
            _bg_step_idx += 1
            _bg_step_timer -= _bg_step_delay
            st = step["t"]
            if st == "init":
                _bg_node_status[step["src"]] = "frontier"
            elif st == "settle":
                _bg_node_status[step["u"]] = "settled"
            elif st == "relax":
                u, v = step["u"], step["v"]
                _bg_settle_edges.add((min(u,v), max(u,v)))
            elif st == "update":
                if _bg_node_status.get(step["v"]) != "settled":
                    _bg_node_status[step["v"]] = "frontier"
            elif st == "path":
                path = step["path"]
                for k in range(len(path)-1):
                    u, v = path[k], path[k+1]
                    _bg_path_edges.add((min(u,v), max(u,v)))
                for nd in path:
                    _bg_node_status[nd] = "path"
                _bg_phase = "path_flash"
                _bg_flash_timer = 0.0
                return
        # If we ran out of steps with no path event
        if _bg_step_idx >= len(_bg_steps):
            _bg_phase = "fade_out"

    elif _bg_phase == "path_flash":
        _bg_flash_timer += dt
        if _bg_flash_timer >= _BG_FLASH_DUR:
            _bg_phase = "fade_out"
            _bg_fade_timer = 0.0

    elif _bg_phase == "fade_out":
        _bg_fade_timer += dt
        if _bg_fade_timer >= _BG_FADE_DUR:
            _bg_restart()


def draw_title_bg(t):
    # Overall opacity envelope — fade in on start, fade out at end
    if _bg_phase == "fade_out":
        global_a = max(0.0, 1.0 - _bg_fade_timer / _BG_FADE_DUR)
    else:
        global_a = 1.0

    # ── Edges ──
    drawn = set()
    for i in range(len(_BG_PTS)):
        nxi, nyi = _BG_PTS[i]
        for j, _ in _BG_ADJ[i]:
            ek = (min(i,j), max(i,j))
            if ek in drawn: continue
            drawn.add(ek)
            nxj, nyj = _BG_PTS[j]
            x1, y1 = int(nxi*SW), int(nyi*SH)
            x2, y2 = int(nxj*SW), int(nyj*SH)

            if ek in _bg_path_edges:
                flash = 0.55 + 0.45 * math.sin(_bg_flash_timer * 4.5)
                col   = c(COL_PATH_OPT, global_a * flash * 0.75)
                thick = 2.5
            elif ek in _bg_settle_edges:
                col   = c(COL_SELECTED, global_a * 0.25)
                thick = 1.5
            else:
                col   = c(COL_BORDER, global_a * 0.10)
                thick = 1.0

            rl.draw_line_ex(rl.Vector2(x1,y1), rl.Vector2(x2,y2), thick, col)

    # ── Nodes ──
    for i, (nx, ny) in enumerate(_BG_PTS):
        x, y = int(nx*SW), int(ny*SH)
        ns   = _bg_node_status.get(i, "default")

        if ns == "path":
            flash = 0.5 + 0.5 * math.sin(_bg_flash_timer * 4.5 + i*0.6)
            fill  = c(COL_PATH_OPT,  global_a * flash * 0.80)
            ring  = c(COL_PATH_OPT,  global_a * flash * 0.30)
            r     = 9
        elif ns == "frontier":
            pulse = 0.5 + 0.5 * math.sin(t * 3.5 + i)
            fill  = c(COL_FRONTIER,  global_a * pulse * 0.65)
            ring  = c(COL_FRONTIER,  global_a * pulse * 0.20)
            r     = 8
        elif ns == "settled":
            fill  = c(COL_SETTLED,   global_a * 0.70)
            ring  = c(COL_BORDER,    global_a * 0.20)
            r     = 7
        else:
            fill  = c(COL_NODE,      global_a * 0.55)
            ring  = c(COL_BORDER,    global_a * 0.18)
            r     = 7

        if ns in ("path", "frontier"):
            rl.draw_circle(x, y, r+5, ring)
        rl.draw_circle(x, y, r, fill)

    # ── Source / dest markers ──
    if _bg_src < len(_BG_PTS):
        sx, sy = int(_BG_PTS[_bg_src][0]*SW), int(_BG_PTS[_bg_src][1]*SH)
        rl.draw_circle(sx, sy, 10, c(COL_START, global_a * 0.55))
        rl.draw_circle_lines(sx, sy, 10, c(COL_START, global_a * 0.80))
    if _bg_dst < len(_BG_PTS):
        dx, dy = int(_BG_PTS[_bg_dst][0]*SW), int(_BG_PTS[_bg_dst][1]*SH)
        rl.draw_circle(dx, dy, 10, c(COL_END, global_a * 0.55))
        rl.draw_circle_lines(dx, dy, 10, c(COL_END, global_a * 0.80))


# ── Title screen ──────────────────────────────────────────────────────────────
def draw_title(mx, my, t):
    draw_title_bg(t)
    rl.draw_circle(SW//2, SH//2-20, int(SH*0.45), c(BG, 0.60))
    cy = SH//2
    ctext("Are you smarter than", SW//2, cy-145, 38, c(GRAY, 0.9))
    ctext("Dijkstra?",            SW//2, cy-94,  72, WHITE)
    uw = int(400*(0.5+0.5*math.sin(t*1.4)))
    ctext("Find the shortest path - beat the algorithm!", SW//2, cy+8, 16, c(GRAY, 0.7))
    bw, bh   = 180, 52
    sbx, sby = SW//2-bw-16, cy+60
    wbx, wby = SW//2+16,    cy+60
    sh = sbx <= mx <= sbx+bw and sby <= my <= sby+bh
    wh = wbx <= mx <= wbx+bw and wby <= my <= wby+bh
    draw_btn(sbx, sby, bw, bh, "Start",  True, sh, 22, COL_ACCENT)
    draw_btn(wbx, wby, bw, bh, "What?",  True, wh, 22, COL_BTN)
    return sh, wh


def draw_what(mx, my):
    for gx in range(0, SW, 60):
        for gy in range(0, SH, 60):
            rl.draw_pixel(gx, gy, c(COL_BORDER, 0.18))
    ctext("What is this?", SW//2, 80, 44, WHITE)
    rl.draw_rectangle(SW//2-80, 134, 160, 3, COL_ACCENT)
    lines = [
        "You'll be shown a map of nodes (circles) connected by edges (lines).",
        "Each edge has a weight — lower weight means a shorter / cheaper path.",
        "",
        "Two nodes are highlighted:  green = Start     red = End.",
        "",
        "Your job: click through nodes to build a path from Start to End",
        "with the lowest total weight you can find.",
        "",
        "Dijkstra's algorithm will calculate the true optimal shortest path.",
        "Can you match it — or beat it?",
        "",
        "You can only pick nodes directly connected to your last pick.",
        "Use  <- Delete  to undo your last choice, then  Submit  when done.",
    ]
    y = 170
    for line in lines:
        if line == "": y += 10; continue
        lw = rl.measure_text(line, 16)
        rl.draw_text(line, SW//2-lw//2, y, 16, c(WHITE, 0.85))
        y += 28
    bw, bh = 160, 48
    bx, by = SW//2-bw//2, SH-90
    bh2 = bx <= mx <= bx+bw and by <= my <= by+bh
    draw_btn(bx, by, bw, bh, "<- Back", True, bh2, 18, COL_BTN)
    return bh2


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
rl.init_window(1200, 800, "Are you smarter than Dijkstra?")
rl.set_target_fps(60)
rl.set_window_state(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
update_layout()

screen      = SCR_TITLE
norm_nodes  = []
edges       = []
n           = 0
start_node  = -1
end_node    = -1
user_path   = []
hover_node  = -1
t           = 0.0

# timing
game_state     = GS_COUNTDOWN
countdown_t    = 0.0      # counts up; show 3 at 0s, 2 at 1s, 1 at 2s, go at 3s
user_timer     = 0.0      # elapsed seconds the user spent picking
user_time_done = 0.0      # snapshot when user submits

# Dijkstra animation state
anim_steps        = []
anim_idx          = 0
anim_timer        = 0.0
anim_node_status  = {}    # i -> "frontier" | "settled" | "path"
anim_active_edge  = None  # (min,max) currently being relaxed
anim_opt_edges    = set() # edges on the found optimal path
anim_opt_path     = []
anim_opt_cost     = 0.0
dijkstra_time     = 0.0   # how long the animation took (wall-clock)
dijkstra_anim_start = 0.0

# results
user_cost     = 0.0
opt_path_set  = set()
user_edge_set = set()


def init_game():
    global norm_nodes, edges, n, start_node, end_node, user_path, game_state
    global countdown_t, user_timer, user_time_done
    global anim_steps, anim_idx, anim_timer, anim_node_status
    global anim_active_edge, anim_opt_edges, anim_opt_path, anim_opt_cost
    global dijkstra_time, dijkstra_anim_start, user_cost, opt_path_set, user_edge_set

    norm_nodes, edges = gen_graph()
    n = len(norm_nodes)
    start_node, end_node = pick_start_end(norm_nodes, edges)
    user_path  = [start_node]
    game_state = GS_COUNTDOWN
    countdown_t = 0.0
    user_timer  = 0.0; user_time_done = 0.0
    anim_steps = []; anim_idx = 0; anim_timer = 0.0
    anim_node_status = {}; anim_active_edge = None
    anim_opt_edges = set(); anim_opt_path = []; anim_opt_cost = 0.0
    dijkstra_time = 0.0; dijkstra_anim_start = 0.0
    user_cost = 0.0; opt_path_set = set(); user_edge_set = set()
    print(f"\n[GAME] New game | Start: {LETTERS[start_node]}  End: {LETTERS[end_node]}")


def start_dijkstra_animation():
    global anim_steps, anim_idx, anim_timer, anim_node_status
    global anim_active_edge, anim_opt_edges, game_state, dijkstra_anim_start
    anim_steps = build_anim_steps(edges, start_node, end_node, n)
    anim_idx   = 0; anim_timer = 0.0
    anim_node_status = {}; anim_active_edge = None; anim_opt_edges = set()
    game_state = GS_ANIMATING
    dijkstra_anim_start = t
    print("[ANIM] Dijkstra animation started.")


def apply_anim_step(step):
    global anim_node_status, anim_active_edge, anim_opt_edges
    global anim_opt_path, anim_opt_cost, game_state, dijkstra_time

    st = step["t"]
    if st == "init":
        anim_node_status[step["src"]] = "frontier"
    elif st == "settle":
        u = step["u"]
        if anim_node_status.get(u) not in ("start_keep", "end_keep"):
            anim_node_status[u] = "settled"
        anim_active_edge = None
    elif st == "relax":
        u, v = step["u"], step["v"]
        anim_active_edge = (min(u,v), max(u,v))
    elif st == "update":
        v = step["v"]
        if anim_node_status.get(v) not in ("settled",):
            anim_node_status[v] = "frontier"
    elif st == "path":
        anim_opt_path = step["path"]
        anim_opt_cost = path_cost(edges, anim_opt_path)
        for k in range(len(anim_opt_path)-1):
            u, v = anim_opt_path[k], anim_opt_path[k+1]
            anim_opt_edges.add((min(u,v), max(u,v)))
        for nd in anim_opt_path:
            anim_node_status[nd] = "path"
        dijkstra_time = round(t - dijkstra_anim_start, 2)
        game_state = GS_RESULTS
        print(f"[ANIM] Done. Optimal: {' -> '.join(LETTERS[i] for i in anim_opt_path)}  cost={anim_opt_cost}")
        print(f"[ANIM] Animation took {dijkstra_time}s wall-clock.")


# ── Main loop ─────────────────────────────────────────────────────────────────
while not rl.window_should_close():
    update_layout()
    dt    = rl.get_frame_time()
    t    += dt
    mx    = rl.get_mouse_x()
    my    = rl.get_mouse_y()
    click = rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT)

    rl.begin_drawing()
    rl.clear_background(BG)

    # ══════════════════════════════════════════════════════════════════════════
    if screen == SCR_TITLE:
        _update_bg(dt)
        sh, wh = draw_title(mx, my, t)
        if click:
            if sh:
                init_game(); screen = SCR_GAME
                print("[NAV] Title -> Game")
            elif wh:
                screen = SCR_WHAT
                print("[NAV] Title -> What")

    # ══════════════════════════════════════════════════════════════════════════
    elif screen == SCR_WHAT:
        back_hov = draw_what(mx, my)
        if click and back_hov:
            screen = SCR_TITLE
            print("[NAV] What -> Title")

    # ══════════════════════════════════════════════════════════════════════════
    elif screen == SCR_GAME:

        # ── tick countdown ───────────────────────────────────────────────────
        if game_state == GS_COUNTDOWN:
            countdown_t += dt
            if countdown_t >= 3.0:
                game_state = GS_BUILDING
                user_timer = 0.0
                print("[GAME] Countdown done — timer started.")

        # ── tick user timer ──────────────────────────────────────────────────
        elif game_state == GS_BUILDING:
            user_timer += dt

        # ── tick Dijkstra animation ──────────────────────────────────────────
        elif game_state == GS_ANIMATING:
            anim_timer += dt
            while anim_timer >= ANIM_STEP_DELAY and anim_idx < len(anim_steps):
                apply_anim_step(anim_steps[anim_idx])
                anim_idx  += 1
                anim_timer -= ANIM_STEP_DELAY
            if anim_idx >= len(anim_steps) and game_state == GS_ANIMATING:
                # no path step found — unlikely but safe
                game_state = GS_RESULTS

        # ── build edge sets for drawing ──────────────────────────────────────
        cur_user_edges = set()
        for k in range(len(user_path)-1):
            u, v = user_path[k], user_path[k+1]
            cur_user_edges.add((min(u,v), max(u,v)))

        # decide draw mode
        if game_state in (GS_COUNTDOWN, GS_BUILDING):
            draw_mode = "building"
        elif game_state == GS_ANIMATING:
            draw_mode = "animating"
        else:
            draw_mode = "results"

        # ── dot grid ─────────────────────────────────────────────────────────
        for gx in range(0, SW, 40):
            for gy in range(0, MAP_H, 40):
                rl.draw_pixel(gx, gy, c(COL_BORDER, 0.18))

        # ── draw edges and nodes ─────────────────────────────────────────────
        # During countdown/building hover only works in building mode
        if game_state == GS_BUILDING:
            hover_node = node_at_px(norm_nodes, mx, my)
        else:
            hover_node = -1

        draw_map_edges(norm_nodes, edges, n,
                       cur_user_edges, anim_opt_edges, anim_active_edge,
                       t, mode=draw_mode)

        draw_map_nodes(norm_nodes, n, start_node, end_node, user_path,
                       hover_node, anim_node_status,
                       set(anim_opt_path), t, mode=draw_mode)

        # ── COUNTDOWN overlay ────────────────────────────────────────────────
        if game_state == GS_COUNTDOWN:
            # grey out the map area
            rl.draw_rectangle(0, 0, SW, MAP_H, c(BG, 0.75))

            num = 3 - int(countdown_t)   # 3, 2, 1
            progress = countdown_t - int(countdown_t)  # 0..1 within each second
            scale_t  = 1.0 - progress * 0.4            # shrinks as second ticks

            if num > 0:
                label = str(num)
                alpha = 1.0 - progress * 0.3
                fs    = int(160 * scale_t)
                lw    = rl.measure_text(label, fs)
                rl.draw_text(label, SW//2 - lw//2, MAP_H//2 - fs//2,
                             fs, c(COL_ACCENT, alpha))
                # ring that shrinks each second
                r = int((1.0 - progress) * 100)
                rl.draw_circle_lines(SW//2, MAP_H//2, r, c(COL_ACCENT, 0.3))
            else:
                # "GO!" flash
                go_alpha = max(0.0, 1.0 - (countdown_t - 3.0) * 4)
                if go_alpha > 0:
                    fs = 140
                    lw = rl.measure_text("GO!", fs)
                    rl.draw_text("GO!", SW//2-lw//2, MAP_H//2-fs//2,
                                 fs, c(COL_START, go_alpha))

        # ── live timer (top-right of map) ─────────────────────────────────────
        if game_state == GS_BUILDING:
            timer_str = fmt_time(user_timer)
            tw = rl.measure_text(timer_str, 22)
            # pill background
            rl.draw_rectangle(SW-tw-28, 46, tw+18, 32, c(PANEL_BG, 0.85))
            rl.draw_rectangle_lines_ex(rl.Rectangle(SW-tw-28, 46, tw+18, 32), 1, COL_ACCENT)
            rl.draw_text(timer_str, SW-tw-18, 53, 22, COL_ACCENT)

        elif game_state in (GS_ANIMATING, GS_RESULTS):
            # show frozen user time
            timer_str = fmt_time(user_time_done)
            tw = rl.measure_text(timer_str, 22)
            rl.draw_rectangle(SW-tw-28, 46, tw+18, 32, c(PANEL_BG, 0.85))
            rl.draw_rectangle_lines_ex(rl.Rectangle(SW-tw-28, 46, tw+18, 32), 1, c(COL_ACCENT, 0.5))
            rl.draw_text(timer_str, SW-tw-18, 53, 22, c(COL_ACCENT, 0.6))

        # ── RESULTS overlay ──────────────────────────────────────────────────
        if game_state == GS_RESULTS:
            user_cost_final = path_cost(edges, user_path)
            opt_cost_final  = path_cost(edges, anim_opt_path)
            won             = user_cost_final <= opt_cost_final

            # semi-transparent overlay card
            card_w = min(SW - 80, 780); card_h = 210
            card_x = SW//2 - card_w//2; card_y = MAP_H//2 - card_h//2 - 20
            rl.draw_rectangle(card_x, card_y, card_w, card_h, c(PANEL_BG, 0.94))
            rl.draw_rectangle_lines_ex(rl.Rectangle(card_x, card_y, card_w, card_h), 2,
                                       COL_ACCENT if won else COL_END)

            # headline
            headline = "You matched Dijkstra!" if user_cost_final == opt_cost_final else \
                       "You beat Dijkstra!  " if won else "Dijkstra wins this round."
            headline_col = COL_START if won else COL_END
            ctext(headline, SW//2, card_y+18, 22, headline_col)

            # divider
            rl.draw_line(card_x+20, card_y+52, card_x+card_w-20, card_y+52, c(PANEL_LINE, 0.8))

            half = card_w//2
            # ── User column ──
            col_ux = card_x + half//2
            ctext("Your Path", col_ux, card_y+62, 14, c(COL_PATH_USR, 0.8))
            user_letters = " -> ".join(LETTERS[i] for i in user_path)
            ctext(user_letters, col_ux, card_y+84, 13, COL_PATH_USR)
            ctext(f"Cost: {user_cost_final}", col_ux, card_y+110, 18,
                  COL_START if won else c(WHITE, 0.85))
            ctext(f"Time: {fmt_time(user_time_done)}", col_ux, card_y+136, 15, c(GRAY, 0.9))

            # vertical divider
            rl.draw_line(card_x+half, card_y+58, card_x+half, card_y+card_h-20,
                         c(PANEL_LINE, 0.6))

            # ── Dijkstra column ──
            col_dx = card_x + half + half//2
            ctext("Dijkstra's Path", col_dx, card_y+62, 14, c(COL_PATH_OPT, 0.8))
            opt_letters = " -> ".join(LETTERS[i] for i in anim_opt_path)
            ctext(opt_letters, col_dx, card_y+84, 13, COL_PATH_OPT)
            ctext(f"Cost: {opt_cost_final}", col_dx, card_y+110, 18,
                  COL_END if not won else c(WHITE, 0.85))
            ctext(f"Steps: {len(anim_steps)}", col_dx, card_y+136, 15, c(GRAY, 0.9))

            # Play again button
            pa_w=180; pa_h=44
            pa_x=SW//2-pa_w//2; pa_y=card_y+card_h-pa_h-8
            pa_hov = pa_x <= mx <= pa_x+pa_w and pa_y <= my <= pa_y+pa_h
            draw_btn(pa_x, pa_y, pa_w, pa_h, "Play Again", True, pa_hov, 18, COL_ACCENT)
            if click and pa_hov:
                init_game()
                print("[NAV] Play Again -> new game")

        # ── title bar ────────────────────────────────────────────────────────
        rl.draw_rectangle(0, 0, SW, 36, c(PANEL_BG, 0.95))
        rl.draw_line(0, 36, SW, 36, PANEL_LINE)
        ctext("Are you smarter than Dijkstra?", SW//2, 9, 17, c(WHITE, 0.65))

        back_w=110; back_h=28
        back_hov = 8 <= mx <= 8+back_w and 4 <= my <= 4+back_h
        draw_btn(8, 4, back_w, back_h, "<- Menu", True, back_hov, 14, COL_BTN)

        # ── bottom panel ─────────────────────────────────────────────────────
        panel_result = draw_panel(start_node, end_node, user_path, game_state,
                                  user_time_done, mx, my)

        # ── click handling ────────────────────────────────────────────────────
        if click:
            # 1. Menu button always takes priority
            if back_hov:
                screen = SCR_TITLE
                print("[NAV] Game -> Title")

            # 2. Node click in the map area — checked BEFORE panel so elif doesn't block it
            elif (game_state == GS_BUILDING
                  and hover_node != -1
                  and my > 36
                  and my < MAP_H
                  and hover_node != user_path[-1]):
                last = user_path[-1]
                if are_adjacent(edges, last, hover_node):
                    if hover_node in user_path and hover_node != end_node:
                        print(f"[PATH]   Skipped {LETTERS[hover_node]} - already in path.")
                    else:
                        user_path.append(hover_node)
                        print(f"[PATH]   Added {LETTERS[hover_node]}  | Path: {[LETTERS[i] for i in user_path]}")
                        if hover_node == end_node:
                            print("[PATH]   End node reached - ready to submit!")
                else:
                    print(f"[PATH]   Skipped {LETTERS[hover_node]} - not adjacent to {LETTERS[last]}.")

            # 3. Panel Submit / Delete buttons
            elif panel_result is not None and game_state == GS_BUILDING:
                (sub_bx, sub_by, btn_w, btn_h, sub_hov, can_submit,
                 del_bx, del_by, del_hov, can_del) = panel_result
                if sub_hov and can_submit:
                    user_time_done = round(user_timer, 2)
                    user_cost      = path_cost(edges, user_path)
                    for k in range(len(user_path)-1):
                        u, v = user_path[k], user_path[k+1]
                        user_edge_set.add((min(u,v), max(u,v)))
                    print(f"\n[SUBMIT] User path: {' -> '.join(LETTERS[i] for i in user_path)}")
                    print(f"[SUBMIT] User cost: {user_cost}  Time: {user_time_done}s")
                    start_dijkstra_animation()
                elif del_hov and can_del:
                    removed = user_path.pop()
                    print(f"[PATH]   Deleted {LETTERS[removed]}  | Path: {[LETTERS[i] for i in user_path]}")


    rl.end_drawing()

rl.close_window()