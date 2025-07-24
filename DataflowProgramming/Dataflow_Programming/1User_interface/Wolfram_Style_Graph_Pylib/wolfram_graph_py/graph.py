"""
WolframGraphPy: A Python library for graph construction inspired by Wolfram Language
"""

from typing import List, Tuple, Dict, Union, Callable, Optional, Any
import itertools
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class Graph:
    """
    A Graph class that mimics Wolfram Language's Graph functionality
    """
    
    def __init__(self, vertices=None, edges=None, directed=False, **kwargs):
        """
        Initialize a Graph object
        
        Args:
            vertices: List of vertices or number of vertices
            edges: List of edges as tuples (u, v) or with weights (u, v, weight)
            directed: Boolean indicating if the graph is directed
            **kwargs: Additional graph properties (layout, style, etc.)
        """
        self.directed = directed
        self.properties = kwargs
        
        # Create the appropriate NetworkX graph
        if directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()
        
        # Add vertices
        self._add_vertices(vertices)
        
        # Add edges
        if edges:
            self._add_edges(edges)
    
    def _add_vertices(self, vertices):
        """Add vertices to the graph"""
        if vertices is None:
            return
        
        if isinstance(vertices, int):
            # Add vertices with names 1 through n
            self.graph.add_nodes_from(range(1, vertices + 1))
        else:
            # Add the specified vertices
            self.graph.add_nodes_from(vertices)
    
    def _add_edges(self, edges):
        """Add edges to the graph"""
        if not edges:
            return
            
        formatted_edges = []
        for edge in edges:
            if isinstance(edge, tuple):
                if len(edge) == 2:
                    # Edge with no weight
                    formatted_edges.append((edge[0], edge[1], {'weight': 1}))
                elif len(edge) == 3:
                    # Edge with weight
                    formatted_edges.append((edge[0], edge[1], {'weight': edge[2]}))
            elif isinstance(edge, list) and len(edge) == 2:
                # Edge as a list instead of tuple
                formatted_edges.append((edge[0], edge[1], {'weight': 1}))
        
        self.graph.add_edges_from(formatted_edges)
    
    def __str__(self):
        """String representation of the graph"""
        return f"Graph with {self.graph.number_of_nodes()} vertices and {self.graph.number_of_edges()} edges"
    
    def plot(self, layout=None, node_size=300, node_color='skyblue', 
             edge_color='black', font_size=10, with_labels=True, **kwargs):
        """
        Plot the graph using matplotlib
        
        Args:
            layout: Layout algorithm ('spring', 'circular', 'random', 'shell', etc.)
            node_size: Size of the nodes
            node_color: Color of the nodes
            edge_color: Color of the edges
            font_size: Size of the vertex labels
            with_labels: Whether to show vertex labels
            **kwargs: Additional plotting parameters
        """
        plt.figure(figsize=kwargs.get('figsize', (8, 6)))
        
        # Determine the layout
        pos = self._get_layout(layout)
        
        # Draw the graph
        nx.draw(self.graph, pos, with_labels=with_labels, 
                node_size=node_size, node_color=node_color,
                edge_color=edge_color, font_size=font_size,
                **{k: v for k, v in kwargs.items() if k != 'figsize'})
        
        plt.axis('off')
        plt.tight_layout()
        
        return plt
    
    def _get_layout(self, layout=None):
        """Get the layout function based on the specified layout"""
        if layout is None:
            layout = self.properties.get('layout', 'spring')
        
        layout_funcs = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'random': nx.random_layout,
            'shell': nx.shell_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'planar': nx.planar_layout,
            'spectral': nx.spectral_layout,
            'spiral': nx.spiral_layout
        }
        
        if layout in layout_funcs:
            return layout_funcs[layout](self.graph)
        else:
            # Default to spring layout
            return nx.spring_layout(self.graph)
    
    def adjacency_matrix(self):
        """Return the adjacency matrix of the graph"""
        return nx.adjacency_matrix(self.graph).toarray()
    
    def degree(self, vertex=None):
        """
        Get the degree of a vertex or all vertices
        
        Args:
            vertex: The vertex to get the degree of, or None for all vertices
        
        Returns:
            The degree of the specified vertex or a dictionary of degrees
        """
        if vertex is not None:
            return self.graph.degree[vertex]
        return dict(self.graph.degree())
    
    def vertices(self):
        """Get the list of vertices in the graph"""
        return list(self.graph.nodes())
    
    def edges(self, with_weights=False):
        """
        Get the list of edges in the graph
        
        Args:
            with_weights: Whether to include edge weights
        
        Returns:
            A list of edges as tuples (u, v) or (u, v, weight)
        """
        if with_weights:
            return [(u, v, d.get('weight', 1)) for u, v, d in self.graph.edges(data=True)]
        return list(self.graph.edges())


# Graph Construction Functions

def CompleteGraph(n):
    """
    Create a complete graph with n vertices
    
    Args:
        n: Number of vertices
    
    Returns:
        A complete graph
    """
    G = Graph(n, directed=False)
    edges = list(itertools.combinations(range(1, n+1), 2))
    G._add_edges(edges)
    return G

def CycleGraph(n):
    """
    Create a cycle graph with n vertices
    
    Args:
        n: Number of vertices
    
    Returns:
        A cycle graph
    """
    G = Graph(n, directed=False)
    edges = [(i, i % n + 1) for i in range(1, n)]
    edges.append((n, 1))
    G._add_edges(edges)
    return G

def PathGraph(n):
    """
    Create a path graph with n vertices
    
    Args:
        n: Number of vertices
    
    Returns:
        A path graph
    """
    G = Graph(n, directed=False)
    edges = [(i, i+1) for i in range(1, n)]
    G._add_edges(edges)
    return G

def StarGraph(n):
    """
    Create a star graph with n vertices
    
    Args:
        n: Number of vertices
    
    Returns:
        A star graph
    """
    G = Graph(n, directed=False)
    edges = [(1, i) for i in range(2, n+1)]
    G._add_edges(edges)
    return G

def WheelGraph(n):
    """
    Create a wheel graph with n vertices
    
    Args:
        n: Number of vertices (including the central vertex)
    
    Returns:
        A wheel graph
    """
    G = Graph(n, directed=False)
    # Connect central vertex to all others
    edges = [(1, i) for i in range(2, n+1)]
    # Connect the outer cycle
    for i in range(2, n):
        edges.append((i, i+1))
    edges.append((n, 2))
    G._add_edges(edges)
    return G

def GridGraph(m, n=None):
    """
    Create a grid graph with m×n vertices
    
    Args:
        m: Number of rows
        n: Number of columns (if None, create an m×m grid)
    
    Returns:
        A grid graph
    """
    if n is None:
        n = m
    
    total_vertices = m * n
    G = Graph(total_vertices, directed=False)
    
    # Add edges
    edges = []
    for i in range(1, m+1):
        for j in range(1, n+1):
            vertex = (i-1)*n + j
            
            # Connect to the right
            if j < n:
                edges.append((vertex, vertex+1))
            
            # Connect downward
            if i < m:
                edges.append((vertex, vertex+n))
    
    G._add_edges(edges)
    G.properties['layout'] = 'grid'
    return G

def RandomGraph(n, p):
    """
    Create a random graph with n vertices and edge probability p
    
    Args:
        n: Number of vertices
        p: Probability of creating an edge between any two vertices
    
    Returns:
        A random graph
    """
    G = Graph(n, directed=False)
    
    # Generate random edges
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            if np.random.random() < p:
                G._add_edges([(i, j)])
                
    return G

def DirectedGraph(edges):
    """
    Create a directed graph from a list of edges
    
    Args:
        edges: List of edges as tuples (from, to) or (from, to, weight)
    
    Returns:
        A directed graph
    """
    # Find all unique vertices in the edges
    vertices = set()
    for edge in edges:
        vertices.add(edge[0])
        vertices.add(edge[1])
    
    G = Graph(list(vertices), edges, directed=True)
    return G

def BipartiteGraph(n1, n2, edges=None):
    """
    Create a bipartite graph with n1 and n2 vertices in each partition
    
    Args:
        n1: Number of vertices in the first partition
        n2: Number of vertices in the second partition
        edges: List of edges between the partitions
    
    Returns:
        A bipartite graph
    """
    G = Graph(n1 + n2, directed=False)
    
    # If edges are not provided, create a complete bipartite graph
    if edges is None:
        edges = [(i, j + n1) for i in range(1, n1+1) for j in range(1, n2+1)]
    
    G._add_edges(edges)
    return G

def CompleteBipartiteGraph(n1, n2):
    """
    Create a complete bipartite graph with n1 and n2 vertices in each partition
    
    Args:
        n1: Number of vertices in the first partition
        n2: Number of vertices in the second partition
    
    Returns:
        A complete bipartite graph
    """
    return BipartiteGraph(n1, n2)

# Graph Operations and Properties

def ShortestPath(G, source, target):
    """
    Find the shortest path between source and target vertices
    
    Args:
        G: Graph object
        source: Source vertex
        target: Target vertex
    
    Returns:
        List of vertices in the shortest path
    """
    try:
        path = nx.shortest_path(G.graph, source=source, target=target)
        return path
    except nx.NetworkXNoPath:
        return None

def ConnectedComponents(G):
    """
    Find the connected components of an undirected graph
    
    Args:
        G: Graph object
    
    Returns:
        List of lists, where each list is a connected component
    """
    if G.directed:
        components = list(nx.weakly_connected_components(G.graph))
    else:
        components = list(nx.connected_components(G.graph))
    
    return [list(component) for component in components]

def IsConnected(G):
    """
    Check if the graph is connected
    
    Args:
        G: Graph object
    
    Returns:
        Boolean indicating if the graph is connected
    """
    if G.directed:
        return nx.is_weakly_connected(G.graph)
    return nx.is_connected(G.graph)

def Distance(G, source, target):
    """
    Find the shortest distance between source and target vertices
    
    Args:
        G: Graph object
        source: Source vertex
        target: Target vertex
    
    Returns:
        The shortest distance or infinity if no path exists
    """
    try:
        return nx.shortest_path_length(G.graph, source=source, target=target)
    except nx.NetworkXNoPath:
        return float('inf')

def GraphUnion(G1, G2, rename=True):
    """
    Create the union of two graphs
    
    Args:
        G1: First Graph object
        G2: Second Graph object
        rename: Boolean indicating whether to automatically rename nodes to avoid conflicts,
                or a tuple of prefixes (prefix1, prefix2) to use for renaming

    Returns:
        A new Graph object representing the union
    """
    if rename is True:
        # Automatically rename nodes to avoid conflicts
        union = nx.disjoint_union(G1.graph, G2.graph)
    elif rename is False:
        # Try to perform the union without renaming (will raise an error if node sets overlap)
        union = nx.union(G1.graph, G2.graph)
    else:
        # Use the provided prefixes for renaming
        union = nx.union(G1.graph, G2.graph, rename=rename)
    
    # Create result graph with the combined directed property
    result = Graph(directed=G1.directed or G2.directed)
    result.graph = union
    
    return result

def GraphProduct(G1, G2, product_type='cartesian'):
    """
    Create the product of two graphs
    
    Args:
        G1: First Graph object
        G2: Second Graph object
        product_type: Type of product ('cartesian', 'tensor', 'strong', 'lexicographic')
    
    Returns:
        A new Graph object representing the product
    """
    product_funcs = {
        'cartesian': nx.cartesian_product,
        'tensor': nx.tensor_product,
        'strong': nx.strong_product,
        'lexicographic': nx.lexicographic_product
    }
    
    if product_type not in product_funcs:
        raise ValueError(f"Unknown product type: {product_type}")
    
    product = product_funcs[product_type](G1.graph, G2.graph)
    
    result = Graph(directed=G1.directed or G2.directed)
    result.graph = product
    
    return result

def GraphComplement(G):
    """
    Create the complement of a graph
    
    Args:
        G: Graph object
    
    Returns:
        A new Graph object representing the complement
    """
    complement = nx.complement(G.graph)
    
    result = Graph(directed=G.directed)
    result.graph = complement
    
    return result

# Specialized Graph Algorithms

def MinimumSpanningTree(G):
    """
    Find the minimum spanning tree of a graph
    
    Args:
        G: Graph object
    
    Returns:
        A new Graph object representing the minimum spanning tree
    """
    if G.directed:
        raise ValueError("Minimum spanning tree is only defined for undirected graphs")
    
    mst = nx.minimum_spanning_tree(G.graph)
    
    result = Graph(directed=False)
    result.graph = mst
    
    return result

def TopologicalSort(G):
    """
    Perform a topological sort on a directed acyclic graph
    
    Args:
        G: Graph object
    
    Returns:
        A list of vertices in topological order
    """
    if not G.directed:
        raise ValueError("Topological sort is only defined for directed graphs")
    
    try:
        return list(nx.topological_sort(G.graph))
    except nx.NetworkXUnfeasible:
        return None  # Graph has cycles

def StronglyConnectedComponents(G):
    """
    Find the strongly connected components of a directed graph
    
    Args:
        G: Graph object
    
    Returns:
        List of lists, where each list is a strongly connected component
    """
    if not G.directed:
        raise ValueError("Strongly connected components are only defined for directed graphs")
    
    components = list(nx.strongly_connected_components(G.graph))
    return [list(component) for component in components]

def PageRank(G, alpha=0.85):
    """
    Calculate PageRank for all vertices
    
    Args:
        G: Graph object
        alpha: Damping parameter
    
    Returns:
        Dictionary of vertices and their PageRank values
    """
    return nx.pagerank(G.graph, alpha=alpha)

def Dijkstra(G, source):
    """
    Find shortest paths from source to all other vertices using Dijkstra's algorithm
    
    Args:
        G: Graph object
        source: Source vertex
    
    Returns:
        Dictionary of shortest distances to all vertices
    """
    return nx.single_source_dijkstra_path_length(G.graph, source)

# Example usage
if __name__ == "__main__":
    # Create a complete graph with 5 vertices
    G1 = CompleteGraph(5)
    print(G1)
    
    # Create a cycle graph with 6 vertices
    G2 = CycleGraph(6)
    print(G2)
    
    # Create a custom graph
    custom_edges = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
    G3 = Graph(5, custom_edges)
    print(G3)
    
    # Plot the graphs
    G1.plot(layout='circular')
    plt.title("Complete Graph K5")
    plt.savefig("complete_graph.png")
    
    G2.plot(layout='circular')
    plt.title("Cycle Graph C6")
    plt.savefig("cycle_graph.png")
    
    G3.plot()
    plt.title("Custom Graph")
    plt.savefig("custom_graph.png")
    
    # Use some algorithms
    print("Is G3 connected?", IsConnected(G3))
    print("Shortest path from 1 to 5 in G3:", ShortestPath(G3, 1, 5))