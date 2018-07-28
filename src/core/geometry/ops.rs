//! Linear algebra operations used by Points, Vectors and Normals.

use num::{abs, Float, Signed};

use super::{PrimFloat, PrimItem, PrimSigned};

/// The length operation
pub trait Length: PrimFloat
where
    Self::Item: PrimItem + Float,
{
    /// Compute the squared length.
    fn length_squared(self) -> Self::Item;

    /// Compute the length.
    fn length(self) -> Self::Item {
        self.length_squared().sqrt()
    }
}

/// The normalise operation
pub trait Normalise: PrimFloat
where
    Self::Item: PrimItem + Float,
{
    /// Compute a normalised (unit) version of a geometric
    /// primitive.
    fn normalise(self) -> Self;
}

/// The dot product operation
pub trait Dot<RHS = Self>: PrimSigned
where
    Self::Item: PrimItem + Signed,
{
    /// Performs the dot product operation.
    fn dot(self, rhs: RHS) -> Self::Item;

    /// Performs the dot product operation and returns the absolute
    /// result.
    fn dot_abs(self, rhs: RHS) -> Self::Item {
        abs(self.dot(rhs))
    }
}

/// The cross product operation
pub trait Cross<RHS = Self>: PrimSigned
where
    Self::Item: PrimItem + Signed,
{
    /// The resulting type after applying the cross product
    /// operation.
    type Output;

    /// Performs the cross product operation.
    fn cross(self, rhs: RHS) -> Self::Output;
}

/// The face-forward operation
pub trait FaceForward<RHS = Self>: PrimSigned
where
    Self::Item: PrimItem + Signed,
{
    /// The resulting type after applying the face-forward
    /// operation.
    type Output;

    /// Performs the face-forward operation.
    fn face_forward(self, rhs: RHS) -> Self::Output;
}
