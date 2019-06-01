//! A two-dimensional point.

use std::convert::From;

use crate::constants::Float;
use crate::geometry::point3::Point3;

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

// From traits

impl From<Point2f> for Point2i {
    fn from(p2f: Point2f) -> Self {
        Point2i::new(p2f.x as i32, p2f.y as i32)
    }
}

impl From<Point2i> for Point2f {
    fn from(p2i: Point2i) -> Self {
        Point2f::new(p2i.x as Float, p2i.y as Float)
    }
}

impl<T> From<Point3<T>> for Point2<T> {
    fn from(p3: Point3<T>) -> Self {
        Point2::new(p3.x, p3.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point3::Point3i;

    use assert_approx_eq::assert_approx_eq;

    fn assert_point2i_equal(p1: Point2i, p2: Point2i) {
        assert_eq!(p1.x, p2.x);
        assert_eq!(p1.y, p2.y);
    }

    fn assert_point2f_equal(p1: Point2f, p2: Point2f) {
        assert_approx_eq!(p1.x, p2.x);
        assert_approx_eq!(p1.y, p2.y);
    }

    // Construction

    #[test]
    fn point2i_new() {
        let given = Point2i::new(0, 1);
        let expected = Point2i { x: 0, y: 1 };
        assert_point2i_equal(given, expected);
    }

    // From traits

    #[test]
    fn point2i_from_point2f() {
        let given = Point2i::from(Point2f::new(0.0, 1.0));
        let expected = Point2i::new(0, 1);
        assert_point2i_equal(given, expected);
    }

    #[test]
    fn point2f_from_point2i() {
        let given = Point2f::from(Point2i::new(0, 1));
        let expected = Point2f::new(0.0, 1.0);
        assert_point2f_equal(given, expected);
    }

    #[test]
    fn point2_from_point3() {
        let given = Point2i::from(Point3i::new(0, 1, 2));
        let expected = Point2i::new(0, 1);
        assert_point2i_equal(given, expected);
    }
}
