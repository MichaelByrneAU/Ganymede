//! A two-dimensional point.

use crate::constants::Float;

/// A two-dimensional point.
#[derive(Debug, Default, Copy, Clone)]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

/// A two-dimensional point of [`i32`]s.
///
/// [`i32`]: https://doc.rust-lang.org/std/primitive.i32.html
pub type Point2i = Point2<i32>;

/// A two-dimensional point of [`Float`]s.
///
/// [`Float`]: ../../constants/type.Float.html
pub type Point2f = Point2<Float>;

impl<T> Point2<T> {
    /// Construct a new [`Point2`] from its components.
    ///
    /// For convenience, use one of the two type aliases, [`Point2i`] or
    /// [`Point2f`], for integer or float versions respectively.
    ///
    /// # Examples
    /// ```
    /// # use ga_core::geometry::point2::{Point2f, Point2i};
    /// let mut p_int = Point2i::new(0, 1);
    /// let mut p_flt = Point2f::new(0.0, 1.0);
    /// ```
    ///
    /// [`Point2`]: struct.Vector2.html
    /// [`Point2i`]: type.Vector2i.html
    /// [`Point2f`]: type.Vector2f.html
    pub fn new(x: T, y: T) -> Self {
        Point2 { x, y }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_point2i_equal(p1: Point2i, p2: Point2i) {
        assert_eq!(p1.x, p2.x);
        assert_eq!(p1.y, p2.y);
    }

    // Construction

    #[test]
    fn point2i_new() {
        let given = Point2i::new(0, 1);
        let expected = Point2i { x: 0, y: 1 };
        assert_point2i_equal(given, expected);
    }
}
