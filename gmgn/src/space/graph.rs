//! Graph space — variable-size graphs with node and optional edge features.
//!
//! Mirrors [Gymnasium `Graph`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Graph).
//!
//! Node and edge feature spaces accept any [`AnySpace`] variant (`Box | Discrete`
//! etc.), matching Gymnasium's polymorphic constraint.
//!
//! # Examples
//!
//! ```
//! use gmgn::space::{GraphSpace, AnySpace, BoundedSpace, Discrete, Space};
//! use gmgn::rng::create_rng;
//!
//! let node_space = AnySpace::from(BoundedSpace::uniform(-1.0, 1.0, 3).unwrap());
//! let edge_space = AnySpace::from(Discrete::new(4));
//! let space = GraphSpace::new(node_space, Some(edge_space));
//! let mut rng = create_rng(Some(42));
//! let sample = space.sample(&mut rng);
//! assert!(space.contains(&sample));
//! ```

use rand::RngExt as _;

use crate::rng::Rng;
use crate::space::dict::{AnySpace, AnyValue};
use crate::space::{Space, SpaceInfo};

/// A single graph instance produced by [`GraphSpace`].
///
/// Node and edge features are [`AnyValue`] to support both continuous
/// (`BoundedSpace`) and discrete (`Discrete`) feature spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphInstance {
    /// Per-node feature values.
    pub nodes: Vec<AnyValue>,
    /// Per-edge feature values (`None` when no edge space).
    pub edges: Option<Vec<AnyValue>>,
    /// Adjacency list — `(src, dst)` pairs indexing into `nodes`.
    pub edge_links: Option<Vec<(usize, usize)>>,
}

/// A space of variable-size graphs with polymorphic node and edge features.
///
/// Accepts any [`AnySpace`] variant for node and edge feature spaces,
/// matching Gymnasium's `Box | Discrete` constraint.
#[derive(Debug, Clone)]
pub struct GraphSpace {
    /// Space describing each node's features.
    node_space: AnySpace,
    /// Space describing each edge's features (`None` = no edge data).
    edge_space: Option<AnySpace>,
}

impl GraphSpace {
    /// Create a new graph space.
    #[must_use]
    pub const fn new(node_space: AnySpace, edge_space: Option<AnySpace>) -> Self {
        Self {
            node_space,
            edge_space,
        }
    }

    /// The node feature space.
    #[must_use]
    pub const fn node_space(&self) -> &AnySpace {
        &self.node_space
    }

    /// The edge feature space, if any.
    #[must_use]
    pub const fn edge_space(&self) -> Option<&AnySpace> {
        self.edge_space.as_ref()
    }
}

impl Space for GraphSpace {
    type Element = GraphInstance;

    fn sample(&self, rng: &mut Rng) -> GraphInstance {
        // Sample a small random graph (1..=10 nodes, 0..=2*nodes edges).
        let num_nodes = rng.random_range(1_usize..=10);
        let nodes: Vec<AnyValue> = (0..num_nodes)
            .map(|_| self.node_space.sample(rng))
            .collect();

        let (edges, edge_links) = self.edge_space.as_ref().map_or((None, None), |es| {
            let max_edges = num_nodes.saturating_mul(2);
            let num_edges = if max_edges == 0 {
                0
            } else {
                rng.random_range(0..=max_edges)
            };
            let edge_vals: Vec<AnyValue> = (0..num_edges).map(|_| es.sample(rng)).collect();
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
        if value.nodes.is_empty() {
            return false;
        }
        if !value.nodes.iter().all(|n| self.node_space.contains(n)) {
            return false;
        }

        let num_nodes = value.nodes.len();

        match (&self.edge_space, &value.edges, &value.edge_links) {
            (None, None, None) => true,
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
    use crate::space::{BoundedSpace, Discrete};

    fn make_space() -> GraphSpace {
        let node_space = AnySpace::from(BoundedSpace::uniform(-1.0, 1.0, 3).unwrap());
        let edge_space = AnySpace::from(Discrete::new(4));
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
        }
    }

    #[test]
    fn discrete_node_space() {
        let node_space = AnySpace::from(Discrete::new(5));
        let space = GraphSpace::new(node_space, None);
        let mut rng = create_rng(Some(0));
        let sample = space.sample(&mut rng);
        assert!(space.contains(&sample));
        assert!(sample.edges.is_none());
    }

    #[test]
    fn no_edge_space() {
        let node_space = AnySpace::from(BoundedSpace::uniform(0.0, 1.0, 2).unwrap());
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
            nodes: vec![AnyValue::Continuous(vec![0.0, 0.0, 0.0])],
            edges: Some(vec![AnyValue::Discrete(0)]),
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
