# Equivalent Resistance Using Graph Theory

## Introduction

Calculating equivalent resistance is a fundamental problem in electrical circuits, essential for understanding and designing efficient systems. While traditional methods involve iteratively applying series and parallel resistor rules, these approaches can become cumbersome for complex circuits with many components. Graph theory offers a powerful alternative, providing a structured and algorithmic way to analyze circuits.

By representing a circuit as a graph—where nodes correspond to junctions and edges represent resistors with weights equal to their resistance values—we can systematically simplify even the most intricate networks. This method streamlines calculations and enables automated analysis, making it particularly useful in circuit simulation software, optimization problems, and network design.

## Algorithm Implementation

The implementation uses NetworkX to represent and manipulate the circuit graph. The main algorithm iteratively simplifies the circuit graph until only the source and target nodes remain connected by a single equivalent resistor.

```python
import networkx as nx
import numpy as np

class EquivalentResistanceCalculator:
    def __init__(self):
        self.graph = None
        self.simplified = False
    
    def load_circuit(self, graph):
        """
        Loads a circuit graph where:
        - Nodes represent junctions
        - Edges represent resistors with resistance as weights
        """
        self.graph = graph.copy()
        self.simplified = False
        return self
    
    def calculate_equivalent_resistance(self, source, target):
        """
        Calculate the equivalent resistance between two nodes in the circuit.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            float: The equivalent resistance between source and target
        """
        if not self.graph:
            raise ValueError("No circuit loaded")
            
        if source not in self.graph or target not in self.graph:
            raise ValueError("Source or target node not in circuit")
        
        # Create a working copy of the graph
        working_graph = self.graph.copy()
        
        # Keep simplifying until we have only the source and target nodes
        while len(working_graph.nodes) > 2 or len(working_graph.edges) > 1:
            # Try to simplify series resistors
            series_simplified = self._simplify_series(working_graph, source, target)
            
            # Try to simplify parallel resistors
            parallel_simplified = self._simplify_parallel(working_graph)
            
            # If no simplification was possible, we need a more advanced technique
            if not (series_simplified or parallel_simplified):
                # Use Y-Δ transformation or node elimination
                self._eliminate_node(working_graph, source, target)
        
        # After simplification, there should be a single edge between source and target
        if len(working_graph.edges) == 1:
            edge_data = list(working_graph.edges(data=True))[0]
            return edge_data[2]['weight']
        else:
            # If there are no edges, the nodes are disconnected
            return float('inf')
```

### Series Reduction

The algorithm identifies nodes with exactly two connections (degree 2) that aren't the source or target nodes. These nodes are intermediate points in series connections and can be eliminated by combining the resistances.

```python
def _simplify_series(self, graph, source, target):
    """
    Identify and simplify series resistors in the circuit.
    A node with exactly two connections can be eliminated.
    """
    simplified = False
    
    # Find nodes that are in series (degree 2 and not source/target)
    series_nodes = [node for node in graph.nodes() 
                  if node != source and node != target and graph.degree(node) == 2]
    
    for node in series_nodes:
        # Get the two adjacent nodes and edge weights
        neighbors = list(graph.neighbors(node))
        if len(neighbors) != 2:
            continue
            
        n1, n2 = neighbors
        
        # Calculate the combined resistance
        r1 = graph[node][n1]['weight']
        r2 = graph[node][n2]['weight']
        combined_resistance = r1 + r2
        
        # Remove the middle node and add a direct edge
        graph.remove_node(node)
        graph.add_edge(n1, n2, weight=combined_resistance)
        
        simplified = True
        break  # We modified the graph, so break the loop
        
    return simplified
```

### Parallel Reduction

The algorithm looks for multiple edges between the same pair of nodes, representing parallel resistors, and combines them according to the parallel resistance formula.

```python
def _simplify_parallel(self, graph):
    """
    Identify and simplify parallel resistors in the circuit.
    Multiple edges between the same nodes are combined.
    """
    simplified = False
    
    # Find parallel edges
    for u in graph.nodes():
        for v in graph.nodes():
            if u >= v:  # Skip repeated pairs and self-loops
                continue
                
            if graph.has_edge(u, v):
                # Get all parallel edges between u and v
                parallel_edges = list(graph.get_edge_data(u, v).items())
                
                if len(parallel_edges) > 1:
                    # Calculate combined resistance (1/R = 1/R1 + 1/R2 + ...)
                    combined_conductance = sum(1/data['weight'] for _, data in parallel_edges)
                    combined_resistance = 1/combined_conductance
                    
                    # Remove all existing edges
                    graph.remove_edge(u, v)
                    
                    # Add new edge with combined resistance
                    graph.add_edge(u, v, weight=combined_resistance)
                    
                    simplified = True
                    break  # We modified the graph, so break the loop
                    
        if simplified:
            break
            
    return simplified
```

### Node Elimination

For complex configurations where simple series or parallel reductions aren't possible, the algorithm uses node elimination. This technique eliminates a non-terminal node by creating new connections between all its neighbors, with appropriate equivalent resistances.

```python
def _eliminate_node(self, graph, source, target):
    """
    Eliminate a node using the node elimination technique.
    Choose a node that's neither source nor target and eliminate it.
    """
    # Find a node to eliminate (not source or target)
    nodes_to_consider = [n for n in graph.nodes() if n != source and n != target]
    
    if not nodes_to_consider:
        return False
        
    node_to_eliminate = nodes_to_consider[0]
    neighbors = list(graph.neighbors(node_to_eliminate))
    
    # Process each pair of neighbors
    for i in range(len(neighbors)):
        for j in range(i+1, len(neighbors)):
            n1, n2 = neighbors[i], neighbors[j]
            
            # Calculate resistance between n1 and n2 through node_to_eliminate
            r1 = graph[node_to_eliminate][n1]['weight']
            r2 = graph[node_to_eliminate][n2]['weight']
            
            # Add new edge or update existing one
            if graph.has_edge(n1, n2):
                existing_r = graph[n1][n2]['weight']
                # Parallel combination of the new path and existing path
                new_r = 1 / (1/existing_r + 1/(r1 + r2))
                graph[n1][n2]['weight'] = new_r
            else:
                # New direct path
                graph.add_edge(n1, n2, weight=r1 + r2)
    
    # Remove the eliminated node
    graph.remove_node(node_to_eliminate)
    return True
```

## Test Examples

### 1. Simple Series Circuit
```
A -- 2Ω -- B -- 3Ω -- C
```

Expected result: 5Ω (2Ω + 3Ω)

```python
def test_simple_series():
    """Test a simple series circuit."""
    circuit = create_circuit([
        ('A', 'B', 2),
        ('B', 'C', 3)
    ])
    calculator = EquivalentResistanceCalculator()
    result = calculator.load_circuit(circuit).calculate_equivalent_resistance('A', 'C')
    
    print(f"Simple Series Circuit: {result}Ω (Expected: 5Ω)")
    return result
```

### 2. Simple Parallel Circuit
```
A -- 6Ω -- B
|         |
+-- 3Ω -- +
```

Expected result: 2Ω (1/R = 1/6 + 1/3)

```python
def test_simple_parallel():
    """Test a simple parallel circuit."""
    circuit = create_circuit([
        ('A', 'B', 6),
        ('A', 'B', 3)
    ])
    calculator = EquivalentResistanceCalculator()
    result = calculator.load_circuit(circuit).calculate_equivalent_resistance('A', 'B')
    
    print(f"Simple Parallel Circuit: {result}Ω (Expected: 2Ω)")
    return result
```

### 3. Complex Circuit
```
     +-- 3Ω -- B -- 1Ω --+
     |                    |
A ---+-- 2Ω -- C -- 4Ω --+--- E
     |                    |
     +-- 5Ω -- D -- 2Ω --+
```

This circuit contains both series and parallel elements, plus mesh configurations. The algorithm handles it by applying node elimination when necessary.

```python
def test_complex_circuit():
    """Test a complex circuit with both series and parallel components."""
    circuit = create_circuit([
        ('A', 'B', 3),
        ('B', 'E', 1),
        ('A', 'C', 2),
        ('C', 'E', 4),
        ('A', 'D', 5),
        ('D', 'E', 2)
    ])
    calculator = EquivalentResistanceCalculator()
    result = calculator.load_circuit(circuit).calculate_equivalent_resistance('A', 'E')
    
    print(f"Complex Circuit: {result}Ω")
    return result
```

## Efficiency Analysis

- **Time Complexity**: O(|V|³), where |V| is the number of nodes. This is dominated by the node elimination step which considers all pairs of neighbors.
- **Space Complexity**: O(|V|² + |E|) to store the graph representation.

## Potential Improvements

1. **Optimized Node Selection**: Instead of arbitrarily picking the first available node for elimination, select nodes based on their degree or other heuristics to minimize computational effort.

2. **Matrix-Based Approach**: For highly complex circuits, implement a more efficient matrix-based approach using modified nodal analysis or the Laplacian matrix of the graph.

3. **Parallelization**: For very large circuits, parallelize certain operations to improve performance.

4. **Smart Detection of Substructures**: Implement advanced pattern recognition to identify complex subcircuits that can be reduced in a single step.

## Conclusion

Graph theory provides an elegant and systematic approach to calculating equivalent resistance in electrical circuits. This implementation demonstrates how series/parallel reductions combined with node elimination can handle arbitrary circuit configurations. The algorithm is robust for practical applications but can be further optimized for very large or complex networks.