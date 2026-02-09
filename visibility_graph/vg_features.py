"""
Visibility Graph Feature Extraction Module

Extracts topological features from Visibility Graphs:
- Degree statistics
- Clustering coefficient
- Power law / exponential fitting
- Motif counts
- Community structure
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, Optional, Tuple, List
import warnings

from .vg_core import (
    construct_vg_fast,
    construct_hvg_fast,
    compute_degree_sequence
)


# =============================================================================
# DEGREE DISTRIBUTION ANALYSIS
# =============================================================================

def compute_degree_stats(adj: np.ndarray) -> Dict[str, float]:
    """
    Compute degree statistics from adjacency matrix.
    
    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix
    
    Returns
    -------
    dict
        Degree statistics
    """
    degrees = adj.sum(axis=1)
    
    return {
        'degree_mean': float(np.mean(degrees)),
        'degree_std': float(np.std(degrees)),
        'degree_max': int(np.max(degrees)),
        'degree_min': int(np.min(degrees)),
        'degree_median': float(np.median(degrees)),
        'degree_skew': float(stats.skew(degrees)),
        'degree_kurtosis': float(stats.kurtosis(degrees))
    }


def fit_power_law(degrees: np.ndarray, x_min: int = 1) -> Dict[str, float]:
    """
    Fit power law distribution to degree sequence.
    
    P(k) ~ k^(-gamma)
    
    Parameters
    ----------
    degrees : np.ndarray
        Degree sequence
    x_min : int
        Minimum degree to consider (to avoid low-degree noise)
    
    Returns
    -------
    dict
        Power law fit parameters
    """
    # Filter degrees
    degrees = degrees[degrees >= x_min]
    
    if len(degrees) < 5:
        return {'gamma': np.nan, 'gamma_error': np.nan, 'ks_stat': np.nan}
    
    # Maximum likelihood estimation for power law
    # gamma = 1 + n / sum(log(k / x_min))
    n = len(degrees)
    gamma = 1 + n / np.sum(np.log(degrees / (x_min - 0.5)))
    gamma_error = (gamma - 1) / np.sqrt(n)
    
    # Kolmogorov-Smirnov test
    try:
        # Create power law CDF
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        empirical_cdf = np.cumsum(counts) / n
        theoretical_cdf = 1 - (unique_degrees / x_min) ** (1 - gamma)
        ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
    except:
        ks_stat = np.nan
    
    return {
        'gamma': float(gamma),
        'gamma_error': float(gamma_error),
        'ks_stat': float(ks_stat)
    }


def fit_exponential(degrees: np.ndarray) -> Dict[str, float]:
    """
    Fit exponential distribution to degree sequence.
    
    P(k) ~ exp(-lambda * k)
    
    Used for HVG analysis.
    
    Returns
    -------
    dict
        Exponential fit parameters
    """
    degrees = degrees[degrees > 0]
    
    if len(degrees) < 5:
        return {'lambda': np.nan, 'lambda_error': np.nan}
    
    # MLE for exponential
    lambda_mle = 1.0 / np.mean(degrees)
    lambda_error = lambda_mle / np.sqrt(len(degrees))
    
    return {
        'lambda': float(lambda_mle),
        'lambda_error': float(lambda_error)
    }


def compute_degree_entropy(degrees: np.ndarray) -> float:
    """
    Compute Shannon entropy of degree distribution.
    
    Higher entropy = more uniform degree distribution
    Lower entropy = more concentrated (e.g., few high-degree hubs)
    """
    # Normalize to probability distribution
    unique, counts = np.unique(degrees, return_counts=True)
    probs = counts / counts.sum()
    
    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    return float(entropy)


# =============================================================================
# CLUSTERING COEFFICIENT
# =============================================================================

def compute_local_clustering(adj: np.ndarray, node: int) -> float:
    """
    Compute local clustering coefficient for a single node.
    
    C_i = 2 * T_i / (k_i * (k_i - 1))
    
    where T_i is number of triangles through node i
    and k_i is degree of node i.
    """
    neighbors = np.where(adj[node] > 0)[0]
    k = len(neighbors)
    
    if k < 2:
        return 0.0
    
    # Count edges between neighbors (triangles)
    triangles = 0
    for i, n1 in enumerate(neighbors):
        for n2 in neighbors[i + 1:]:
            if adj[n1, n2] > 0:
                triangles += 1
    
    return 2.0 * triangles / (k * (k - 1))


def compute_clustering_coefficient(adj: np.ndarray) -> Dict[str, float]:
    """
    Compute global and average local clustering coefficient.
    
    Returns
    -------
    dict
        Clustering metrics
    """
    n = adj.shape[0]
    
    # Local clustering coefficients
    local_cc = np.array([compute_local_clustering(adj, i) for i in range(n)])
    
    # Average local clustering (Watts-Strogatz style)
    avg_local_cc = np.mean(local_cc)
    
    # Global clustering (transitivity)
    # C = 3 * (# triangles) / (# connected triples)
    triangles = 0
    triples = 0
    
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        triples += k * (k - 1) // 2
        
        for j_idx, j in enumerate(neighbors):
            for k_node in neighbors[j_idx + 1:]:
                if adj[j, k_node] > 0:
                    triangles += 1
    
    global_cc = 3 * triangles / triples if triples > 0 else 0.0
    
    return {
        'clustering_global': float(global_cc),
        'clustering_avg_local': float(avg_local_cc),
        'clustering_std': float(np.std(local_cc)),
        'clustering_max': float(np.max(local_cc)),
        'clustering_min': float(np.min(local_cc[local_cc > 0])) if np.any(local_cc > 0) else 0.0
    }


# =============================================================================
# MOTIF COUNTING (3-node subgraphs)
# =============================================================================

def count_triads(adj: np.ndarray) -> Dict[str, int]:
    """
    Count different types of 3-node subgraphs (triads).
    
    For undirected graphs:
    - Empty: no edges (we don't count these)
    - One edge: exactly one edge
    - Two edges: path (chain)
    - Three edges: triangle (clique)
    
    Returns
    -------
    dict
        Triad counts
    """
    n = adj.shape[0]
    
    triangles = 0
    paths = 0  # Two edges, not connected
    single_edges = 0  # One edge only
    
    # For each triple of nodes
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                edges = adj[i, j] + adj[j, k] + adj[i, k]
                
                if edges == 3:
                    triangles += 1
                elif edges == 2:
                    paths += 1
                elif edges == 1:
                    single_edges += 1
    
    return {
        'triangles': int(triangles),
        'paths': int(paths),
        'single_edges': int(single_edges),
        'triangle_ratio': float(triangles / (triangles + paths + 1e-10))
    }


def compute_motif_frequencies(adj: np.ndarray) -> Dict[str, float]:
    """
    Compute normalized motif frequencies.
    
    Compares to random graph expectation.
    """
    n = adj.shape[0]
    m = adj.sum() // 2  # Number of edges
    p = 2 * m / (n * (n - 1))  # Edge probability
    
    triads = count_triads(adj)
    
    # Expected number of triangles in random graph with same density
    n_triples = n * (n - 1) * (n - 2) // 6
    expected_triangles = n_triples * (p ** 3)
    
    # Z-score for triangle count
    triangle_zscore = (triads['triangles'] - expected_triangles) / np.sqrt(expected_triangles + 1)
    
    return {
        **triads,
        'expected_triangles': float(expected_triangles),
        'triangle_zscore': float(triangle_zscore)
    }


# =============================================================================
# PATH LENGTH AND SMALL-WORLD METRICS
# =============================================================================

def compute_average_path_length(adj: np.ndarray) -> float:
    """
    Compute average shortest path length using BFS.
    
    For disconnected graphs, only considers connected pairs.
    """
    n = adj.shape[0]
    total_length = 0
    num_pairs = 0
    
    for source in range(n):
        # BFS from source
        distances = np.full(n, -1, dtype=np.int32)
        distances[source] = 0
        queue = [source]
        
        while queue:
            current = queue.pop(0)
            neighbors = np.where(adj[current] > 0)[0]
            
            for neighbor in neighbors:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        
        # Sum distances (excluding -1 for disconnected)
        connected = distances[distances > 0]
        total_length += connected.sum()
        num_pairs += len(connected)
    
    return float(total_length / num_pairs) if num_pairs > 0 else float('inf')


def compute_diameter(adj: np.ndarray) -> int:
    """Compute graph diameter (longest shortest path)."""
    n = adj.shape[0]
    max_dist = 0
    
    for source in range(n):
        distances = np.full(n, -1, dtype=np.int32)
        distances[source] = 0
        queue = [source]
        
        while queue:
            current = queue.pop(0)
            neighbors = np.where(adj[current] > 0)[0]
            
            for neighbor in neighbors:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        
        max_dist = max(max_dist, distances.max())
    
    return int(max_dist)


# =============================================================================
# ASSORTATIVITY
# =============================================================================

def compute_assortativity(adj: np.ndarray) -> float:
    """
    Compute degree assortativity coefficient.
    
    Positive: high-degree nodes connect to high-degree
    Negative: high-degree nodes connect to low-degree (disassortative)
    """
    degrees = adj.sum(axis=1)
    edges = np.array(np.where(np.triu(adj) > 0)).T
    
    if len(edges) == 0:
        return 0.0
    
    # Degree of source and target for each edge
    source_degrees = degrees[edges[:, 0]]
    target_degrees = degrees[edges[:, 1]]
    
    # Pearson correlation
    r, _ = stats.pearsonr(source_degrees, target_degrees)
    
    return float(r) if not np.isnan(r) else 0.0


# =============================================================================
# CENTRALITY MEASURES
# =============================================================================

def compute_centrality_stats(adj: np.ndarray) -> Dict[str, float]:
    """
    Compute centrality statistics.
    
    Uses degree centrality (simple but fast).
    """
    n = adj.shape[0]
    degrees = adj.sum(axis=1)
    
    # Degree centrality (normalized)
    degree_centrality = degrees / (n - 1)
    
    # Centralization (how centralized is the network)
    max_centrality = degree_centrality.max()
    sum_diff = np.sum(max_centrality - degree_centrality)
    max_sum_diff = (n - 1) * (n - 2) / (n - 1)  # For star graph
    centralization = sum_diff / max_sum_diff if max_sum_diff > 0 else 0
    
    return {
        'centrality_mean': float(np.mean(degree_centrality)),
        'centrality_std': float(np.std(degree_centrality)),
        'centrality_max': float(max_centrality),
        'network_centralization': float(centralization)
    }


# =============================================================================
# MAIN FEATURE EXTRACTION
# =============================================================================

def extract_graph_features(
    adj: np.ndarray,
    include_motifs: bool = True,
    include_path_length: bool = False  # Expensive for large graphs
) -> Dict[str, float]:
    """
    Extract comprehensive features from a visibility graph.
    
    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix
    include_motifs : bool
        Whether to compute motif counts (slower)
    include_path_length : bool
        Whether to compute path length (slowest)
    
    Returns
    -------
    dict
        Dictionary of graph features
    """
    features = {}
    
    # Basic graph properties
    n = adj.shape[0]
    m = adj.sum() // 2
    features['num_nodes'] = n
    features['num_edges'] = int(m)
    features['density'] = float(2 * m / (n * (n - 1)))
    
    # Degree statistics
    degrees = compute_degree_sequence(adj)
    features.update(compute_degree_stats(adj))
    
    # Degree distribution fitting
    features.update(fit_power_law(degrees))
    features.update(fit_exponential(degrees))
    features['degree_entropy'] = compute_degree_entropy(degrees)
    
    # Clustering
    features.update(compute_clustering_coefficient(adj))
    
    # Assortativity
    features['assortativity'] = compute_assortativity(adj)
    
    # Centrality
    features.update(compute_centrality_stats(adj))
    
    # Motifs (optional - slower)
    if include_motifs:
        features.update(compute_motif_frequencies(adj))
    
    # Path length (optional - slowest)
    if include_path_length:
        features['avg_path_length'] = compute_average_path_length(adj)
        features['diameter'] = compute_diameter(adj)
    
    return features


# =============================================================================
# ROLLING WINDOW FEATURE EXTRACTION
# =============================================================================

def compute_rolling_vg_features(
    series: np.ndarray,
    window_size: int = 20,
    step_size: int = 1,
    use_hvg: bool = True,
    include_motifs: bool = False,
    verbose: bool = False
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Compute VG features on a rolling window.
    
    IMPORTANT: No lookahead bias - features at time t only use data up to time t.
    
    Parameters
    ----------
    series : np.ndarray
        1D time series
    window_size : int
        Size of rolling window
    step_size : int
        Step between windows (1 = every point, 2 = every other point, etc.)
    use_hvg : bool
        If True, use HVG (faster). If False, use VG.
    include_motifs : bool
        Whether to compute motif features
    verbose : bool
        Print progress
    
    Returns
    -------
    Tuple[np.ndarray, List[Dict]]
        (time_indices, features_list)
        time_indices: indices where features are computed (first valid is window_size - 1)
        features_list: list of feature dictionaries
    """
    n = len(series)
    
    if window_size > n:
        raise ValueError(f"Window size {window_size} > series length {n}")
    
    time_indices = []
    features_list = []
    
    # Rolling window (no lookahead)
    for end_idx in range(window_size - 1, n, step_size):
        start_idx = end_idx - window_size + 1
        window = series[start_idx:end_idx + 1]
        
        if verbose and (end_idx - window_size + 1) % 100 == 0:
            print(f"Processing window ending at {end_idx}/{n}")
        
        # Construct graph
        if use_hvg:
            adj = construct_hvg_fast(window)
        else:
            adj = construct_vg_fast(window)
        
        # Extract features
        features = extract_graph_features(
            adj,
            include_motifs=include_motifs,
            include_path_length=False  # Too slow for rolling
        )
        
        time_indices.append(end_idx)
        features_list.append(features)
    
    return np.array(time_indices), features_list


def features_to_dataframe(
    time_indices: np.ndarray,
    features_list: List[Dict[str, float]],
    index: Optional[np.ndarray] = None
) -> 'pd.DataFrame':
    """
    Convert feature list to pandas DataFrame.
    
    Parameters
    ----------
    time_indices : np.ndarray
        Time indices for each feature row
    features_list : list
        List of feature dictionaries
    index : np.ndarray, optional
        Original index (e.g., dates) to use
    
    Returns
    -------
    pd.DataFrame
    """
    import pandas as pd
    
    df = pd.DataFrame(features_list)
    
    if index is not None:
        df.index = index[time_indices]
    else:
        df['time_idx'] = time_indices
    
    return df
