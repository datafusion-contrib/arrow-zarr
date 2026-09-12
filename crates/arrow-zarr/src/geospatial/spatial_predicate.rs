// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use datafusion::physical_plan::PhysicalExpr;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpatialRelationType {
    Within,
    Contains,
    Intersects,
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
            "st_intersects" => Some(Self::Intersects),
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
            // Intersects is symmetric, so swapping operands leaves it unchanged.
            Self::Intersects => Self::Intersects,
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
            SpatialRelationType::from_name("st_intersects"),
            Some(SpatialRelationType::Intersects)
        );

        assert_eq!(
            SpatialRelationType::Within.opposite(),
            SpatialRelationType::Contains
        );
        assert_eq!(
            SpatialRelationType::Contains.opposite(),
            SpatialRelationType::Within
        );
        assert_eq!(
            SpatialRelationType::Intersects.opposite(),
            SpatialRelationType::Intersects
        );
    }
}
