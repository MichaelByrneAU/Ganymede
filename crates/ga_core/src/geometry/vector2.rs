//! A two-dimensional vector.

use std::ops::{Index, IndexMut};

use crate::constants::Float;

/// A two-dimensional vector.
#[derive(Debug, Default, Copy, Clone)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

/// A two-dimensional vector of [`i32`]s.
///
/// [`i32`]: https://doc.rust-lang.org/std/primitive.i32.html
pub type Vector2i = Vector2<i32>;

/// A two-dimensional vector of [`Float`]s.
///
/// [`Float`]: ../../constants/type.Float.html
pub type Vector2f = Vector2<Float>;

impl<T> Vector2<T> {
    /// Construct a new [`Vector2`] from its components.
    ///
    /// For convenience, use one of the two type aliases, [`Vector2i`] or
    /// [`Vector2f`], for integer or float versions respectively.
    ///
    /// # Examples
    /// ```
    /// # use ga_core::geometry::vector2::{Vector2f, Vector2i};
    /// let mut v_int = Vector2i::new(0, 1);
    /// let mut v_flt = Vector2f::new(0.0, 1.0);
    /// ```
    ///
    /// [`Vector2`]: struct.Vector2.html
    /// [`Vector2i`]: type.Vector2i.html
    /// [`Vector2f`]: type.Vector2f.html
    pub fn new(x: T, y: T) -> Self {
        Vector2 { x, y }
    }
}

impl Vector2f {
    /// Check whether any component holds a NaN value.
    pub fn has_nans(&self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }
}

// Indexing traits

impl<T> Index<usize> for Vector2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("out of bounds access (Vector2)"),
        }
    }
}

impl<T> IndexMut<usize> for Vector2<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("out of bounds access (Vector2)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    fn assert_vector2i_equal(v1: Vector2i, v2: Vector2i) {
        assert_eq!(v1.x, v2.x);
        assert_eq!(v1.y, v2.y);
    }

    fn assert_vector2f_equal(v1: Vector2f, v2: Vector2f) {
        assert_approx_eq!(v1.x, v2.x);
        assert_approx_eq!(v1.y, v2.y);
    }

    // Construction

    #[test]
    fn vector2i_new() {
        let given = Vector2i::new(0, 1);
        let expected = Vector2i { x: 0, y: 1 };
        assert_vector2i_equal(given, expected);
    }

    #[test]
    fn vector2f_new() {
        let given = Vector2f::new(0.0, 1.0);
        let expected = Vector2f { x: 0.0, y: 1.0 };
        assert_vector2f_equal(given, expected);
    }

    // NaN Checking
    #[test]
    fn vector2f_has_nans() {
        let given = Vector2f::new(0.0, 1.0);
        assert!(!given.has_nans());
        let given = Vector2f::new(0.0 / 0.0, 1.0);
        assert!(given.has_nans());
    }

    // Indexing traits

    #[test]
    fn vector2_index() {
        let given = Vector2i::new(1, 2);
        assert_eq!(given[0], 1);
        assert_eq!(given[1], 2);
    }

    #[test]
    #[should_panic]
    fn vector2_index_out_of_bounds() {
        let _ = Vector2i::new(1, 2)[2];
    }

    #[test]
    fn vector2_index_mut() {
        let mut given = Vector2i::new(1, 2);
        given[0] = 2;
        given[1] = 3;
        assert_eq!(given[0], 2);
        assert_eq!(given[1], 3);
    }

    #[test]
    #[should_panic]
    fn vector2_index_mut_out_of_bounds() {
        let mut given = Vector2i::new(1, 2);
        given[2] = 3;
    }
}
