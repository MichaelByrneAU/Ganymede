//! A three-dimensional point.

use crate::constants::Float;

/// A three-dimensional point.
#[derive(Debug, Default, Copy, Clone)]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// A three-dimensional point of [`i32`]s.
///
/// [`i32`]: https://doc.rust-lang.org/std/primitive.i32.html
pub type Point3i = Point3<i32>;

/// A three-dimensional point of [`Float`]s.
///
/// [`Float`]: ../../constants/type.Float.html
pub type Point3f = Point3<Float>;

impl<T> Point3<T> {
    /// Construct a new [`Point3`] from its components.
    ///
    /// For convenience, use one of the two type aliases, [`Point3i`] or
    /// [`Point3f`], for integer or float versions respectively.
    ///
    /// # Examples
    /// ```
    /// # use ga_core::geometry::point3::{Point3f, Point3i};
    /// let mut p_int = Point3i::new(0, 1, 2);
    /// let mut p_flt = Point3f::new(0.0, 1.0, 2.0);
    /// ```
    ///
    /// [`Point3`]: struct.Vector2.html
    /// [`Point3i`]: type.Vector2i.html
    /// [`Point3f`]: type.Vector2f.html
    pub fn new(x: T, y: T, z: T) -> Self {
        Point3 { x, y, z }
    }
}

// From traits

impl From<Point3f> for Point3i {
    fn from(p3f: Point3f) -> Self {
        Point3i::new(p3f.x as i32, p3f.y as i32, p3f.z as i32)
    }
}

impl From<Point3i> for Point3f {
    fn from(p3i: Point3i) -> Self {
        Point3f::new(p3i.x as Float, p3i.y as Float, p3i.z as Float)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use assert_approx_eq::assert_approx_eq;

    fn assert_point3i_equal(p1: Point3i, p2: Point3i) {
        assert_eq!(p1.x, p2.x);
        assert_eq!(p1.y, p2.y);
        assert_eq!(p1.z, p2.z);
    }

    fn assert_point3f_equal(p1: Point3f, p2: Point3f) {
        assert_approx_eq!(p1.x, p2.x);
        assert_approx_eq!(p1.y, p2.y);
        assert_approx_eq!(p1.z, p2.z);
    }

    // Construction

    #[test]
    fn point3i_new() {
        let given = Point3i::new(0, 1, 2);
        let expected = Point3i { x: 0, y: 1, z: 2 };
        assert_point3i_equal(given, expected);
    }

    // From traits

    #[test]
    fn point3i_from_point3f() {
        let given = Point3i::from(Point3f::new(0.0, 1.0, 2.0));
        let expected = Point3i::new(0, 1, 2);
        assert_point3i_equal(given, expected);
    }

    #[test]
    fn point3f_from_point3i() {
        let given = Point3f::from(Point3i::new(0, 1, 2));
        let expected = Point3f::new(0.0, 1.0, 2.0);
        assert_point3f_equal(given, expected);
    }
}
