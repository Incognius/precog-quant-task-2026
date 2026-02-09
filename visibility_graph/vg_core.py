"""
Visibility Graph Core Module

Implements efficient construction of:
- Visibility Graph (VG)
- Horizontal Visibility Graph (HVG)

References:
- Lacasa et al. (2008): "From time series to complex networks: The visibility graph"
- Luque et al. (2009): "Horizontal visibility graphs: Exact results for random time series"
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional
import warnings


# =============================================================================
# NAIVE IMPLEMENTATIONS (for validation)
# =============================================================================

def construct_visibility_graph(
    series: np.ndarray,
    return_adjacency: bool = True
) -> np.ndarray:
    """
    Construct a Visibility Graph from a time series.
    
    Two nodes (i, j) are connected if for all k in (i, j):
        y[k] < y[j] + (y[i] - y[j]) * (j - k) / (j - i)
    
    Parameters
    ----------
    series : np.ndarray
        1D time series of shape (n,)
    return_adjacency : bool
        If True, return adjacency matrix. If False, return edge list.
    
    Returns
    -------
    np.ndarray
        If return_adjacency: (n, n) adjacency matrix
        If not: (m, 2) edge list where m is number of edges
    
    Notes
    -----
    Time complexity: O(n^2) naive, O(n log n) with divide-and-conquer
    """
    n = len(series)
    
    if return_adjacency:
        adj = np.zeros((n, n), dtype=np.int8)
    else:
        edges = []
    
    for i in range(n):
        for j in range(i + 2, n):  # j > i + 1 (adjacent nodes always connected)
            visible = True
            for k in range(i + 1, j):
                # Visibility condition
                threshold = series[j] + (series[i] - series[j]) * (j - k) / (j - i)
                if series[k] >= threshold:
                    visible = False
                    break
            
            if visible:
                if return_adjacency:
                    adj[i, j] = 1
                    adj[j, i] = 1
                else:
                    edges.append((i, j))
        
        # Adjacent nodes are always connected
        if i < n - 1:
            if return_adjacency:
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
            else:
                edges.append((i, i + 1))
    
    if return_adjacency:
        return adj
    else:
        return np.array(edges)


def construct_horizontal_visibility_graph(
    series: np.ndarray,
    return_adjacency: bool = True
) -> np.ndarray:
    """
    Construct a Horizontal Visibility Graph from a time series.
    
    Two nodes (i, j) are connected if for all k in (i, j):
        y[k] < min(y[i], y[j])
    
    This is faster and simpler than the standard VG.
    
    Parameters
    ----------
    series : np.ndarray
        1D time series of shape (n,)
    return_adjacency : bool
        If True, return adjacency matrix. If False, return edge list.
    
    Returns
    -------
    np.ndarray
        Adjacency matrix or edge list
    """
    n = len(series)
    
    if return_adjacency:
        adj = np.zeros((n, n), dtype=np.int8)
    else:
        edges = []
    
    for i in range(n):
        for j in range(i + 2, n):
            threshold = min(series[i], series[j])
            visible = True
            for k in range(i + 1, j):
                if series[k] >= threshold:
                    visible = False
                    break
            
            if visible:
                if return_adjacency:
                    adj[i, j] = 1
                    adj[j, i] = 1
                else:
                    edges.append((i, j))
        
        # Adjacent nodes always connected
        if i < n - 1:
            if return_adjacency:
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
            else:
                edges.append((i, i + 1))
    
    if return_adjacency:
        return adj
    else:
        return np.array(edges)


# =============================================================================
# OPTIMIZED NUMBA IMPLEMENTATIONS
# =============================================================================

@njit(cache=True)
def _vg_visibility_check(series: np.ndarray, i: int, j: int) -> bool:
    """Check if nodes i and j are visible in VG."""
    if j <= i + 1:
        return True  # Adjacent nodes always connected
    
    y_i, y_j = series[i], series[j]
    slope = (y_i - y_j) / (j - i)
    
    for k in range(i + 1, j):
        threshold = y_j + slope * (j - k)
        if series[k] >= threshold:
            return False
    return True


@njit(cache=True)
def _hvg_visibility_check(series: np.ndarray, i: int, j: int) -> bool:
    """Check if nodes i and j are visible in HVG."""
    if j <= i + 1:
        return True
    
    threshold = min(series[i], series[j])
    for k in range(i + 1, j):
        if series[k] >= threshold:
            return False
    return True


@njit(cache=True, parallel=True)
def construct_vg_fast(series: np.ndarray) -> np.ndarray:
    """
    Fast VG construction using Numba with parallel processing.
    
    Parameters
    ----------
    series : np.ndarray
        1D time series
    
    Returns
    -------
    np.ndarray
        (n, n) adjacency matrix
    """
    n = len(series)
    adj = np.zeros((n, n), dtype=np.int8)
    
    for i in prange(n):
        for j in range(i + 1, n):
            if _vg_visibility_check(series, i, j):
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj


@njit(cache=True, parallel=True)
def construct_hvg_fast(series: np.ndarray) -> np.ndarray:
    """
    Fast HVG construction using Numba with parallel processing.
    
    Parameters
    ----------
    series : np.ndarray
        1D time series
    
    Returns
    -------
    np.ndarray
        (n, n) adjacency matrix
    """
    n = len(series)
    adj = np.zeros((n, n), dtype=np.int8)
    
    for i in prange(n):
        for j in range(i + 1, n):
            if _hvg_visibility_check(series, i, j):
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj


@njit(cache=True)
def construct_vg_edge_list(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct VG and return as edge list (more memory efficient).
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (edge_index, edge_count) where edge_index is (max_edges, 2)
        and edge_count is the actual number of edges.
    """
    n = len(series)
    max_edges = n * (n - 1) // 2
    edges = np.zeros((max_edges, 2), dtype=np.int32)
    edge_count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if _vg_visibility_check(series, i, j):
                edges[edge_count, 0] = i
                edges[edge_count, 1] = j
                edge_count += 1
    
    return edges[:edge_count], edge_count


@njit(cache=True)
def construct_hvg_edge_list(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct HVG and return as edge list.
    """
    n = len(series)
    max_edges = n * (n - 1) // 2
    edges = np.zeros((max_edges, 2), dtype=np.int32)
    edge_count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if _hvg_visibility_check(series, i, j):
                edges[edge_count, 0] = i
                edges[edge_count, 1] = j
                edge_count += 1
    
    return edges[:edge_count], edge_count


# =============================================================================
# DIVIDE AND CONQUER VG (O(n log n) for certain series)
# =============================================================================

def construct_vg_divide_conquer(series: np.ndarray) -> np.ndarray:
    """
    Divide-and-conquer VG construction.
    
    For convex hulls of time series, this achieves O(n log n).
    For general series, worst case is still O(n^2).
    
    This implementation uses a recursive approach based on finding
    the maximum element and connecting it to all visible elements.
    """
    n = len(series)
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Mark adjacent nodes
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    
    def _dc_vg(left: int, right: int):
        if right - left <= 1:
            return
        
        # Find maximum in range
        max_idx = left + np.argmax(series[left:right + 1])
        
        # Connect maximum to all nodes on left and right
        # that it can see (using convex hull property)
        
        # Left side
        for i in range(left, max_idx):
            if _vg_visibility_check_py(series, i, max_idx):
                adj[i, max_idx] = 1
                adj[max_idx, i] = 1
        
        # Right side
        for j in range(max_idx + 1, right + 1):
            if _vg_visibility_check_py(series, max_idx, j):
                adj[max_idx, j] = 1
                adj[j, max_idx] = 1
        
        # Recurse
        _dc_vg(left, max_idx)
        _dc_vg(max_idx, right)
    
    _dc_vg(0, n - 1)
    return adj


def _vg_visibility_check_py(series: np.ndarray, i: int, j: int) -> bool:
    """Python version of visibility check (for divide-conquer)."""
    if j <= i + 1:
        return True
    
    y_i, y_j = series[i], series[j]
    slope = (y_i - y_j) / (j - i)
    
    for k in range(i + 1, j):
        threshold = y_j + slope * (j - k)
        if series[k] >= threshold:
            return False
    return True


# =============================================================================
# WEIGHTED VISIBILITY GRAPH
# =============================================================================

@njit(cache=True)
def construct_weighted_vg(series: np.ndarray) -> np.ndarray:
    """
    Construct a weighted VG where edge weight is the slope angle.
    
    This preserves more information than binary VG.
    
    Weight = arctan((y[j] - y[i]) / (j - i))
    
    Returns
    -------
    np.ndarray
        (n, n) weighted adjacency matrix
    """
    n = len(series)
    adj = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(i + 1, n):
            if _vg_visibility_check(series, i, j):
                # Weight is the slope angle
                weight = np.arctan2(series[j] - series[i], j - i)
                adj[i, j] = weight
                adj[j, i] = weight
    
    return adj


# =============================================================================
# DIRECTED VISIBILITY GRAPH
# =============================================================================

@njit(cache=True)
def construct_directed_vg(series: np.ndarray) -> np.ndarray:
    """
    Construct a directed VG.
    
    Edge direction indicates time flow (i -> j means i precedes j).
    This is useful for temporal analysis.
    
    Returns
    -------
    np.ndarray
        (n, n) directed adjacency matrix (asymmetric)
    """
    n = len(series)
    adj = np.zeros((n, n), dtype=np.int8)
    
    for i in range(n):
        for j in range(i + 1, n):
            if _vg_visibility_check(series, i, j):
                adj[i, j] = 1  # Only forward direction
    
    return adj


# =============================================================================
# UTILITIES
# =============================================================================

def adjacency_to_edge_index(adj: np.ndarray) -> np.ndarray:
    """
    Convert adjacency matrix to edge index format for PyTorch Geometric.
    
    Parameters
    ----------
    adj : np.ndarray
        (n, n) adjacency matrix
    
    Returns
    -------
    np.ndarray
        (2, m) edge index where m is number of edges
    """
    edges = np.array(np.where(adj > 0))
    return edges


def edge_list_to_edge_index(edge_list: np.ndarray) -> np.ndarray:
    """
    Convert edge list to bidirectional edge index for GNN.
    
    Parameters
    ----------
    edge_list : np.ndarray
        (m, 2) edge list
    
    Returns
    -------
    np.ndarray
        (2, 2m) edge index with both directions
    """
    # Add reverse edges for undirected graph
    reverse = edge_list[:, ::-1]
    all_edges = np.vstack([edge_list, reverse])
    return all_edges.T


def compute_degree_sequence(adj: np.ndarray) -> np.ndarray:
    """Compute degree sequence from adjacency matrix."""
    return adj.sum(axis=1)


def compute_adjacency_spectrum(adj: np.ndarray) -> np.ndarray:
    """Compute eigenvalues of adjacency matrix."""
    return np.linalg.eigvalsh(adj.astype(np.float64))


# =============================================================================
# VALIDATION
# =============================================================================

def validate_vg_properties(adj: np.ndarray) -> dict:
    """
    Validate that the graph has expected VG properties.
    
    Returns
    -------
    dict
        Validation results
    """
    n = adj.shape[0]
    
    # Check symmetry
    is_symmetric = np.allclose(adj, adj.T)
    
    # Check connectivity (adjacent nodes must be connected)
    adjacent_connected = all(adj[i, i+1] == 1 for i in range(n-1))
    
    # Check no self-loops
    no_self_loops = np.trace(adj) == 0
    
    # Compute basic stats
    num_edges = adj.sum() // 2
    avg_degree = adj.sum(axis=1).mean()
    
    return {
        'is_symmetric': is_symmetric,
        'adjacent_connected': adjacent_connected,
        'no_self_loops': no_self_loops,
        'num_nodes': n,
        'num_edges': int(num_edges),
        'avg_degree': float(avg_degree),
        'is_valid': is_symmetric and adjacent_connected and no_self_loops
    }
