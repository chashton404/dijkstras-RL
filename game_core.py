"""
game_core.py
Shared graph logic — imported by both the RL environment and the raylib viewer.
No raylib dependency here.
"""
import math
import random
import heapq

N_NODES  = 12
LETTERS  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
REF_W    = 1200.0
REF_H    = 576.0


# ── Graph generation ──────────────────────────────────────────────────────────

def gen_graph(n_nodes=N_NODES, seed=None):
    """
    Return (norm_nodes, edges).
      norm_nodes[i] = (nx, ny)  — normalised 0..1 coordinates
      edges[i]      = [(j, weight), ...]
    """
    rng = random.Random(seed) if seed is not None else random
    pad, min_sep = 0.08, 0.13
    nodes, attempts = [], 0
    while len(nodes) < n_nodes and attempts < 8000:
        attempts += 1
        nx = rng.uniform(pad, 1.0 - pad)
        ny = rng.uniform(pad, 1.0 - pad)
        if all(math.hypot(nx - ex, ny - ey) > min_sep for ex, ey in nodes):
            nodes.append((nx, ny))

    n = len(nodes)
    edges = [[] for _ in range(n)]

    def add_edge(i, j):
        nxi, nyi = nodes[i]; nxj, nyj = nodes[j]
        w = round(math.hypot(nxi * REF_W - nxj * REF_W,
                              nyi * REF_H - nyj * REF_H) / 40.0, 1)
        if not any(nb == j for nb, _ in edges[i]):
            edges[i].append((j, w))
            edges[j].append((i, w))

    # Spanning tree — guarantees connectivity
    unvisited = list(range(1, n))
    visited   = [0]
    while unvisited:
        i = rng.choice(visited)
        j = rng.choice(unvisited)
        add_edge(i, j)
        visited.append(j)
        unvisited.remove(j)

    # Extra edges for density
    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(nodes[i][0] - nodes[j][0],
                          nodes[i][1] - nodes[j][1]) < 0.38:
                if rng.random() < 0.32:
                    add_edge(i, j)

    return nodes, edges


# ── Graph algorithms ──────────────────────────────────────────────────────────

def dijkstra_dist(edges, src, dst, n):
    dist = [float("inf")] * n
    dist[src] = 0
    heap = [(0, src)]
    settled = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        if u == dst:
            return d
        for v, w in edges[u]:
            if v not in settled and d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    return float("inf")


def dijkstra_path(edges, src, dst, n):
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[src] = 0
    heap = [(0, src)]
    settled = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in settled:
            continue
        settled.add(u)
        if u == dst:
            break
        for v, w in edges[u]:
            if v not in settled and d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))
    path = []
    cur = dst
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if path and path[0] == src:
        return path, round(dist[dst], 1)
    return [], float("inf")


def pick_start_end(norm_nodes, edges, rng=None):
    """Pick a non-adjacent start/end pair with a valid path between them."""
    if rng is None:
        rng = random
    n = len(norm_nodes)
    adj = set()
    for i in range(n):
        for j, _ in edges[i]:
            adj.add((min(i, j), max(i, j)))
    for _ in range(500):
        s = rng.randint(0, n - 1)
        e = rng.randint(0, n - 1)
        if s == e:
            continue
        if (min(s, e), max(s, e)) in adj:
            continue
        if dijkstra_dist(edges, s, e, n) < float("inf"):
            return s, e
    return 0, n - 1


def path_cost(edges, path):
    total = 0.0
    for k in range(len(path) - 1):
        u, v = path[k], path[k + 1]
        found = next((w for nb, w in edges[u] if nb == v), None)
        if found is None:
            return float("inf")
        total += found
    return round(total, 1)


def are_adjacent(edges, u, v):
    return any(nb == v for nb, _ in edges[u])


def get_neighbors(edges, u):
    return [nb for nb, _ in edges[u]]


def adj_matrix(edges, n):
    """Return an n×n weight matrix (0 = no edge)."""
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j, w in edges[i]:
            mat[i][j] = w
    return mat
