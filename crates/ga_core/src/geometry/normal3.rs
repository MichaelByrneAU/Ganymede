//! A three-dimensional normal

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::constants::Float;
use crate::geometry::vector3::Vector3f;

/// A three-dimensional normal.
#[derive(Debug, Default, Copy, Clone)]
pub struct Normal3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// A three-dimensional normal of [`Float`]s.
///
/// [`Float`]: ../../constants/type.Float.html
pub type Normal3f = Normal3<Float>;

impl<T> Normal3<T> {
    /// Construct a new [`Normal3`] from its components.
    ///
    /// For convenience, use the type alias [`Normal3i`] a float
    /// version.
    ///
    /// # Examples
    /// ```
    /// # use ga_core::geometry::normal3::Normal3f;
    /// let mut n_flt = Normal3f::new(0.0, 1.0, 2.0);
    /// ```
    ///
    /// [`Normal3`]: struct.Normal3.html
    /// [`Normal3i`]: type.Normal3i.html
    pub fn new(x: T, y: T, z: T) -> Self {
        Normal3 { x, y, z }
    }
}

// From traits

impl From<Vector3f> for Normal3f {
    fn from(v3f: Vector3f) -> Self {
        Normal3f::new(v3f.x, v3f.y, v3f.z)
    }
}

// Indexing traits

impl<T> Index<usize> for Normal3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("out of bounds access (Normal3)"),
        }
    }
}

impl<T> IndexMut<usize> for Normal3<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("out of bounds access (Normal3)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::vector3::Vector3f;

    use assert_approx_eq::assert_approx_eq;

    fn assert_normal3f_equal(n1: Normal3f, n2: Normal3f) {
        assert_approx_eq!(n1.x, n2.x);
        assert_approx_eq!(n1.y, n2.y);
        assert_approx_eq!(n1.z, n2.z);
    }

    // Construction

    #[test]
    fn normal3f_new() {
        let given = Normal3f::new(0.0, 1.0, 2.0);
        let expected = Normal3f {
            x: 0.0,
            y: 1.0,
            z: 2.0,
        };
        assert_normal3f_equal(given, expected);
    }

    // From traits

    #[test]
    fn normal3f_from_vector3f() {
        let given = Normal3f::from(Vector3f::new(0.0, 1.0, 2.0));
        let expected = Normal3f::new(0.0, 1.0, 2.0);
        assert_normal3f_equal(given, expected);
    }

    // Indexing traits

    #[test]
    fn normal3_index() {
        let given = Normal3f::new(1.0, 2.0, 3.0);
        assert_eq!(given[0], 1.0);
        assert_eq!(given[1], 2.0);
        assert_eq!(given[2], 3.0);
    }

    #[test]
    #[should_panic]
    fn normal3_index_out_of_bounds() {
        let _ = Normal3f::new(1.0, 2.0, 3.0)[3];
    }

    #[test]
    fn normal3_index_mut() {
        let mut given = Normal3f::new(1.0, 2.0, 3.0);
        given[0] = 2.0;
        given[1] = 3.0;
        given[2] = 4.0;
        assert_normal3f_equal(given, Normal3f::new(2.0, 3.0, 4.0));
    }

    #[test]
    #[should_panic]
    fn normal3_index_mut_out_of_bounds() {
        let mut given = Normal3f::new(1.0, 2.0, 3.0);
        given[3] = 4.0;
    }
}
