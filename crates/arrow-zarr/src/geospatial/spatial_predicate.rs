use std::sync::Arc;

use datafusion::physical_plan::PhysicalExpr;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpatialRelationType {
    Within,
}

#[derive(Debug, Clone)]
pub struct RelationPredicate {
    pub left: Arc<dyn PhysicalExpr>,
    pub right: Arc<dyn PhysicalExpr>,
    pub relation_type: SpatialRelationType,
}

impl SpatialRelationType {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "st_within" => Some(Self::Within),
            _ => None,
        }
    }
}

impl RelationPredicate {
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        right: Arc<dyn PhysicalExpr>,
        relation_type: SpatialRelationType,
    ) -> Self {
        Self {
            left,
            right,
            relation_type,
        }
    }
}
