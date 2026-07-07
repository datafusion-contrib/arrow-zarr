use std::sync::Arc;

use datafusion::physical_plan::PhysicalExpr;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpatialRelationType {
    Within,
    Contains,
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
            "st_contains" => Some(Self::Contains),
            _ => None,
        }
    }

    // The relation with its operands swapped, e.g. a within b is the same as
    // b contains a. Used to express a predicate whose arguments reference the
    // join sides in reverse by swapping the arguments.
    pub fn opposite(&self) -> Self {
        match self {
            Self::Within => Self::Contains,
            Self::Contains => Self::Within,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_name_and_opposite() {
        assert_eq!(
            SpatialRelationType::from_name("st_within"),
            Some(SpatialRelationType::Within)
        );
        assert_eq!(
            SpatialRelationType::from_name("ST_Contains"),
            Some(SpatialRelationType::Contains)
        );

        assert_eq!(
            SpatialRelationType::Within.opposite(),
            SpatialRelationType::Contains
        );
        assert_eq!(
            SpatialRelationType::Contains.opposite(),
            SpatialRelationType::Within
        );
    }
}
