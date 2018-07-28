//! Geometric primitives and associated operations.

use noisy_float::prelude::*;
use num::{Float, Signed};

pub mod normal;
pub mod ops;
pub mod point;
pub mod vector;

/// Behaviours implemented by all geometric primitive items.
pub trait PrimItem: Copy + Clone + Default + Eq + Ord {}

// Attach traits to the base integer and float sizes.
impl PrimItem for isize {}
impl PrimItem for R64 {}

/// Behaviours implemented by all geometric primitives.
pub trait Prim: Sized {
    type Item: PrimItem;

    /// Return the value of the smallest component.
    fn min_component(self) -> Self::Item;

    /// Return the value of the largest component.
    fn max_component(self) -> Self::Item;

    /// Return the index of the smallest component.
    fn min_dimension(self) -> usize;

    /// Return the index of the largest component.
    fn max_dimension(self) -> usize;

    /// Component-wise minimum operation with another primitive.
    fn min(self, other: Self) -> Self;

    /// Component-wise maximum operation with another primitive.
    fn max(self, other: Self) -> Self;
}

/// Behaviours implemented by geometric primitives that contain
/// signed numbers.
pub trait PrimSigned: Prim
where
    Self::Item: PrimItem + Signed,
{
    /// Return a version of the primitive with the absolute value
    /// of all its components.
    fn abs(self) -> Self;
}

/// Behaviours implemented by geometric primitives that contain
/// floats.
pub trait PrimFloat: Prim
where
    Self::Item: PrimItem + Float,
{
    /// Compute the component-wise floor of a primitive.
    fn floor(self) -> Self;

    /// Compute the component-wise ceiling of a primitive.
    fn ceil(self) -> Self;
}
