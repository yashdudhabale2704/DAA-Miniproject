import time
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import tempfile
import streamlit as st

class BellmanFord:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.dist = {v: float("inf") for v in vertices}
        self.parent = {v: None for v in vertices}
        self.memo = {}

    def run(self, source):
        if source in self.memo:
            st.info(f"Using cached results for source '{source}'")
            self.dist = self.memo[source]["dist"].copy()
            self.parent = self.memo[source]["parent"].copy()
            return self.memo[source]["frames"]

        st.write(f"**Computing shortest paths for source `{source}`...**")
        start = time.time()

        self.dist = {v: float("inf") for v in self.vertices}
        self.parent = {v: None for v in self.vertices}
        self.dist[source] = 0
        frames = []

        for i in range(len(self.vertices) - 1):
            updated = False
            for u, v, w in self.edges:
                if self.dist[u] + w < self.dist[v]:
                    self.dist[v] = self.dist[u] + w
                    self.parent[v] = u
                    updated = True
                frames.append({
                    "iteration": i + 1,
                    "u": u, "v": v,
                    "dist": self.dist.copy(),
                    "parent": self.parent.copy()
                })
            if not updated:
                break

        for u, v, w in self.edges:
            if self.dist[u] + w < self.dist[v]:
                st.error("Graph contains a negative weight cycle!")
                return None

        st.success(f"Computation completed in {time.time()-start:.6f}s")
        self.memo[source] = {
            "dist": self.dist.copy(),
            "parent": self.parent.copy(),
            "frames": frames
        }
        return frames

    def get_path(self, target):
        path = []
        while target is not None:
            path.insert(0, target)
            target = self.parent[target]
        return path

    def animate_frames(self, frames, source):
        """Render animation to a temporary GIF file and return its path."""
        G = nx.DiGraph()
        G.add_weighted_edges_from(self.edges)
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(6, 5))

        def update(frame):
            ax.clear()
            iteration = frame["iteration"]
            u, v = frame["u"], frame["v"]
            dist = frame["dist"]
            parent = frame["parent"]

            nx.draw(G, pos, ax=ax, with_labels=True,
                    node_color="lightblue", node_size=1200,
                    font_size=10, arrows=True)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrows=True)

            if (u, v) in G.edges:
                nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)],
                                       edge_color="red", width=2.5, arrows=True)

            tree_edges = [(parent[x], x) for x in self.vertices if parent[x] is not None]
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=tree_edges,
                                   edge_color="green", width=2.5, arrows=True)

            edge_labels = {(a, b): w for a, b, w in self.edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

            labels = {v: f"{v}\n({dist[v] if dist[v]!=float('inf') else '∞'})"
                      for v in self.vertices}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

            ax.set_title(f"Iteration {iteration}: Relaxing edge ({u} → {v})")
            ax.axis("off")

        anim = FuncAnimation(fig, update, frames=frames, interval=800, repeat=False)

        # ✅ Save to a temporary file because BytesIO isn't supported by PillowWriter
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            writer = PillowWriter(fps=1)
            anim.save(tmp.name, writer=writer)
            gif_path = tmp.name

        plt.close(fig)
        return gif_path

    def visualize_final(self, source, target=None):
        """Static visualization of final results."""
        G = nx.DiGraph()
        G.add_weighted_edges_from(self.edges)
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(6, 5))
        nx.draw(G, pos, ax=ax, with_labels=True, node_color="lightblue",
                node_size=1200, font_size=10, arrows=True)
        edge_labels = {(u, v): w for u, v, w in self.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        tree_edges = [(self.parent[v], v) for v in self.vertices if self.parent[v] is not None]
        nx.draw_networkx_edges(G, pos, edgelist=tree_edges, edge_color="green", width=2.5)

        if target:
            path = self.get_path(target)
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=3)
            st.info(f"Shortest path {source} → {target}: {' -> '.join(path)}")

        ax.set_title(f"Final Shortest Paths from {source}")
        ax.axis("off")
        return fig



# =====================================================
# Streamlit Interface
# =====================================================
st.title("Optimal Route Planner")

st.markdown("""
This app demonstrates how the **Bellman–Ford algorithm** computes shortest paths step-by-step,  
with **memoization** to avoid repeated computation.
""")

# Default graph example
default_vertices = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
default_edges = [
    ('A', 'B', 6),
    ('A', 'C', 5),
    ('A', 'D', 5),
    ('C', 'B', -2),
    ('D', 'C', -2),
    ('B', 'E', -1),
    ('C', 'E', -1),
    ('D', 'F', -1),
    ('E', 'G', 3),
    ('F', 'G', 3)
]

# Sidebar input
st.sidebar.header("Graph Configuration")
vertex_input = st.sidebar.text_input("Vertices (comma-separated):", ",".join(default_vertices))
edge_input = st.sidebar.text_area(
    "Edges (format: u,v,w per line):",
    "\n".join([f"{u},{v},{w}" for u, v, w in default_edges])
)

source = st.sidebar.text_input("Source vertex:", "A")
target = st.sidebar.text_input("Target vertex:", "G")

if st.sidebar.button("find the optimal route"):
    try:
        vertices = [v.strip() for v in vertex_input.split(",") if v.strip()]
        edges = []
        for line in edge_input.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) == 3:
                u, v, w = parts
                edges.append((u.strip(), v.strip(), float(w)))

        bf = BellmanFord(vertices, edges)
        frames = bf.run(source)

        if frames:
            st.subheader("Step-by-Step computation")
            gif_path = bf.animate_frames(frames, source)
            st.image(gif_path, caption="Relaxation steps of Bellman–Ford", use_container_width=True)


            st.subheader("Final Shortest Paths")
            fig = bf.visualize_final(source, target)
            st.pyplot(fig)

            st.subheader(" Shortest Distances")
            for v in vertices:
                st.write(f"**{v}** : {bf.dist[v]}")

    except Exception as e:
        st.error(f"Error: {e}")
