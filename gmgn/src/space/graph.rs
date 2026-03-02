//! Graph space — variable-size graphs with node and optional edge features.
//!
//! Mirrors [Gymnasium `Graph`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Graph).
//!
//! Elements are [`GraphInstance`] structs containing node features, optional
//! edge features, and an adjacency list (`edge_links`).
//!
//! # Examples
//!
//! ```
//! use gmgn::space::{GraphSpace, BoundedSpace, Discrete, Space};
//! use gmgn::rng::create_rng;
//!
//! let node_space = BoundedSpace::uniform(-1.0, 1.0, 3).unwrap();
//! let edge_space = Discrete::new(4);
//! let space = GraphSpace::new(node_space, Some(edge_space));
//! let mut rng = create_rng(Some(42));
//! let sample = space.sample(&mut rng);
//! assert!(space.contains(&sample));
//! ```

use rand::RngExt as _;

use crate::rng::Rng;
use crate::space::{BoundedSpace, Discrete, Space, SpaceInfo};

/// A single graph instance produced by [`GraphSpace`].
#[derive(Debug, Clone, PartialEq)]
pub struct GraphInstance {
    /// Node feature matrix — one `Vec<f32>` per node.
    pub nodes: Vec<Vec<f32>>,
    /// Edge feature vector — one `i64` per edge (from a [`Discrete`] edge space).
    /// `None` when the graph has no edge features.
    pub edges: Option<Vec<i64>>,
    /// Adjacency list — `(src, dst)` pairs indexing into `nodes`.
    /// `None` when there are no edges.
    pub edge_links: Option<Vec<(usize, usize)>>,
}

/// A space of variable-size graphs with typed node and optional edge features.
///
/// Node features are drawn from a [`BoundedSpace`] and edge features from a
/// [`Discrete`] space (matching Gymnasium's `Box | Discrete` constraint).
#[derive(Debug, Clone)]
pub struct GraphSpace {
    /// Space describing each node's feature vector.
    node_space: BoundedSpace,
    /// Space describing each edge's feature (`None` = edges carry no data).
    edge_space: Option<Discrete>,
}

impl GraphSpace {
    /// Create a new graph space.
    #[must_use]
    pub const fn new(node_space: BoundedSpace, edge_space: Option<Discrete>) -> Self {
        Self {
            node_space,
            edge_space,
        }
    }

    /// The node feature space.
    #[must_use]
    pub const fn node_space(&self) -> &BoundedSpace {
        &self.node_space
    }

    /// The edge feature space, if any.
    #[must_use]
    pub const fn edge_space(&self) -> Option<&Discrete> {
        self.edge_space.as_ref()
    }
}

impl Space for GraphSpace {
    type Element = GraphInstance;

    fn sample(&self, rng: &mut Rng) -> GraphInstance {
        // Sample a small random graph (1..=10 nodes, 0..=2*nodes edges).
        let num_nodes = rng.random_range(1_usize..=10);
        let nodes: Vec<Vec<f32>> = (0..num_nodes)
            .map(|_| self.node_space.sample(rng))
            .collect();

        let (edges, edge_links) = self.edge_space.as_ref().map_or((None, None), |es| {
            let max_edges = num_nodes.saturating_mul(2);
            let num_edges = if max_edges == 0 {
                0
            } else {
                rng.random_range(0..=max_edges)
            };
            let edge_vals: Vec<i64> = (0..num_edges).map(|_| es.sample(rng)).collect();
            let links: Vec<(usize, usize)> = (0..num_edges)
                .map(|_| {
                    (
                        rng.random_range(0..num_nodes),
                        rng.random_range(0..num_nodes),
                    )
                })
                .collect();
            (Some(edge_vals), Some(links))
        });

        GraphInstance {
            nodes,
            edges,
            edge_links,
        }
    }

    fn contains(&self, value: &GraphInstance) -> bool {
        // All node features must belong to the node space.
        if value.nodes.is_empty() {
            return false;
        }
        if !value.nodes.iter().all(|n| self.node_space.contains(n)) {
            return false;
        }

        let num_nodes = value.nodes.len();

        match (&self.edge_space, &value.edges, &value.edge_links) {
            // No edge space — edges and links should be None.
            (None, None, None) => true,
            // Edge space present — edges and links must both be present and consistent.
            (Some(es), Some(edges), Some(links)) => {
                edges.len() == links.len()
                    && edges.iter().all(|e| es.contains(e))
                    && links
                        .iter()
                        .all(|&(src, dst)| src < num_nodes && dst < num_nodes)
            }
            _ => false,
        }
    }

    fn shape(&self) -> &[usize] {
        &[]
    }

    fn flatdim(&self) -> usize {
        // Variable-size — return node feature dimension as a hint.
        self.node_space.flatdim()
    }

    fn is_flattenable(&self) -> bool {
        false
    }

    fn space_info(&self) -> SpaceInfo {
        SpaceInfo::Graph {
            node_space: Box::new(self.node_space.space_info()),
            edge_space: self.edge_space.as_ref().map(|s| Box::new(s.space_info())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::create_rng;

    fn make_space() -> GraphSpace {
        let node_space = BoundedSpace::uniform(-1.0, 1.0, 3).unwrap();
        let edge_space = Discrete::new(4);
        GraphSpace::new(node_space, Some(edge_space))
    }

    #[test]
    fn sample_and_contains() {
        let space = make_space();
        let mut rng = create_rng(Some(42));
        for _ in 0..20 {
            let sample = space.sample(&mut rng);
            assert!(space.contains(&sample), "sample not in space");
            assert!(!sample.nodes.is_empty());
            for node in &sample.nodes {
                assert_eq!(node.len(), 3);
            }
        }
    }

    #[test]
    fn no_edge_space() {
        let node_space = BoundedSpace::uniform(0.0, 1.0, 2).unwrap();
        let space = GraphSpace::new(node_space, None);
        let mut rng = create_rng(Some(0));
        let sample = space.sample(&mut rng);
        assert!(space.contains(&sample));
        assert!(sample.edges.is_none());
        assert!(sample.edge_links.is_none());
    }

    #[test]
    fn rejects_empty_nodes() {
        let space = make_space();
        let bad = GraphInstance {
            nodes: vec![],
            edges: None,
            edge_links: None,
        };
        assert!(!space.contains(&bad));
    }

    #[test]
    fn rejects_bad_edge_link() {
        let space = make_space();
        let bad = GraphInstance {
            nodes: vec![vec![0.0, 0.0, 0.0]],
            edges: Some(vec![0]),
            edge_links: Some(vec![(0, 5)]), // 5 >= num_nodes
        };
        assert!(!space.contains(&bad));
    }

    #[test]
    fn not_flattenable() {
        let space = make_space();
        assert!(!space.is_flattenable());
    }

    #[test]
    fn space_info_is_graph() {
        let space = make_space();
        assert!(matches!(space.space_info(), SpaceInfo::Graph { .. }));
    }
}
