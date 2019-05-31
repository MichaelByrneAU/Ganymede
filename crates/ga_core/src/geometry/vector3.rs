//! A three-dimensional vector.

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use num::Signed;

use crate::constants::Float;

/// A three-dimensional vector.
#[derive(Debug, Default, Copy, Clone)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// A three-dimensional vector of [`i32`]s.
///
/// [`i32`]: https://doc.rust-lang.org/std/primitive.i32.html
pub type Vector3i = Vector3<i32>;

/// A three-dimensional vector of [`Float`]s.
///
/// [`Float`]: ../../constants/type.Float.html
pub type Vector3f = Vector3<Float>;

impl<T> Vector3<T> {
    /// Construct a new [`Vector3`] from its components.
    ///
    /// For convenience, use one of the two type aliases, [`Vector3i`] or
    /// [`Vector3f`], for integer or float versions respectively.
    ///
    /// # Examples
    /// ```
    /// # use ga_core::geometry::vector3::{Vector3f, Vector3i};
    /// let mut v_int = Vector3i::new(0, 1, 2);
    /// let mut v_flt = Vector3f::new(0.0, 1.0, 2.0);
    /// ```
    ///
    /// [`Vector3`]: struct.Vector3.html
    /// [`Vector3i`]: type.Vector3i.html
    /// [`Vector3f`]: type.Vector3f.html
    pub fn new(x: T, y: T, z: T) -> Self {
        Vector3 { x, y, z }
    }
}

// Methods

impl<T> Vector3<T> {
    /// Return a new [`Vector3`] with absolute values of its components.
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn abs(&self) -> Self
    where
        T: Signed,
    {
        Vector3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Return the squared length of the [`Vector3`].
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn length_squared(&self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Return the minimum component of the [`Vector3`].
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn min_component(&self) -> T
    where
        T: Copy + PartialOrd,
    {
        if self.x < self.y && self.x < self.z {
            self.x
        } else if self.y < self.x && self.y < self.z {
            self.y
        } else {
            self.z
        }
    }

    /// Return the maximum component of the [`Vector3`].
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn max_component(&self) -> T
    where
        T: Copy + PartialOrd,
    {
        if self.x > self.y && self.x > self.z {
            self.x
        } else if self.y > self.x && self.y > self.z {
            self.y
        } else {
            self.z
        }
    }

    /// Return the index corresponding to the minimum component of the [`Vector3`].
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn min_dimension(&self) -> usize
    where
        T: Copy + PartialOrd,
    {
        if self.x < self.y && self.x < self.z {
            0
        } else if self.y < self.x && self.y < self.z {
            1
        } else {
            2
        }
    }

    /// Return the index corresponding to the maximum component of the [`Vector3`].
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn max_dimension(&self) -> usize
    where
        T: Copy + PartialOrd,
    {
        if self.x > self.y && self.x > self.z {
            0
        } else if self.y > self.x && self.y > self.z {
            1
        } else {
            2
        }
    }

    /// Return a new [`Vector3`] with the coordinates permuted according to the indices
    /// provided.
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn permute(&self, x: usize, y: usize, z: usize) -> Self
    where
        T: Copy,
    {
        Vector3::new(self[x], self[y], self[z])
    }
}

impl Vector3<Float> {
    /// Check whether any component holds a NaN value.
    pub fn has_nans(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    /// Return the length of the [`Vector3`].
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn length(&self) -> Float {
        self.length_squared().sqrt()
    }

    /// Return a normalised version of a [`Vector3`].
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn normalise(&self) -> Self {
        *self / self.length()
    }
}

// Associated functions

impl<T> Vector3<T> {
    /// Compute the dot product of two [`Vector3`]s.
    ///
    /// # Examples
    /// ```
    /// # use ga_core::geometry::vector3::{Vector3, Vector3i};
    /// let v1 = Vector3i::new(0, 1, 2);
    /// let v2 = Vector3i::new(1, 2, 3);
    /// assert_eq!(Vector3::dot(v1, v2), 8);
    /// ```
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn dot(v1: Self, v2: Self) -> T
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
    }

    /// Compute the absolute dot product of two [`Vector3`]s.
    ///
    /// # Examples
    /// ```
    /// # use ga_core::geometry::vector3::{Vector3, Vector3i};
    /// let v1 = Vector3i::new(-1, -1, -1);
    /// let v2 = Vector3i::new(1, 2, 3);
    /// assert_eq!(Vector3::dot_abs(v1, v2), 6);
    /// ```
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn dot_abs(v1: Self, v2: Self) -> T
    where
        T: Signed + Add<Output = T> + Mul<Output = T>,
    {
        Vector3::dot(v1, v2).abs()
    }

    /// Compute the cross product of two [`Vector3`]s.
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn cross(v1: Self, v2: Self) -> Self
    where
        T: Copy + Signed + Sub<Output = T> + Mul<Output = T>,
    {
        Vector3::new(
            v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x,
        )
    }
}

impl Vector3<Float> {
    /// Create a coordinate system from a [`Vector3`].
    ///
    /// This assumes that `v1` has already been normalised.
    ///
    /// [`Vector3`]: struct.Vector3.html
    pub fn coordinate_system(v1: Self) -> (Self, Self, Self) {
        let v2 = if v1.x.abs() > v1.y.abs() {
            Vector3::new(-v1.z, 0.0, v1.x) / (v1.x * v1.x + v1.z * v1.z).sqrt()
        } else {
            Vector3::new(0.0, v1.z, -v1.y) / (v1.y * v1.y + v1.z * v1.z).sqrt()
        };
        let v3 = Vector3::cross(v1, v2);
        (v1, v2, v3)
    }
}

// Indexing traits

impl<T> Index<usize> for Vector3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("out of bounds access (Vector3)"),
        }
    }
}

impl<T> IndexMut<usize> for Vector3<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("out of bounds access (Vector3)"),
        }
    }
}

// Negation trait

impl<T> Neg for Vector3<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self {
        Vector3::new(-self.x, -self.y, -self.z)
    }
}

// Addition traits

impl<T> Add for Vector3<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Vector3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl<T> AddAssign for Vector3<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

// Subtraction traits

impl<T> Sub for Vector3<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Vector3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T> SubAssign for Vector3<T>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

// Multiplication traits

impl<T> Mul<T> for Vector3<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Vector3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vector3i> for i32 {
    type Output = Vector3i;

    fn mul(self, rhs: Vector3i) -> Vector3i {
        Vector3i::new(rhs.x * self, rhs.y * self, rhs.z * self)
    }
}

impl Mul<Vector3f> for Float {
    type Output = Vector3f;

    fn mul(self, rhs: Vector3f) -> Vector3f {
        Vector3f::new(rhs.x * self, rhs.y * self, rhs.z * self)
    }
}

impl<T> MulAssign<T> for Vector3<T>
where
    T: Copy + MulAssign<T>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

// Division traits

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<Float> for Vector3f {
    type Output = Self;

    fn div(self, rhs: Float) -> Self {
        debug_assert!(rhs != 0.0);
        let inv = 1.0 / rhs;

        Vector3f::new(self.x * inv, self.y * inv, self.z * inv)
    }
}

impl DivAssign<Float> for Vector3f {
    fn div_assign(&mut self, rhs: Float) {
        debug_assert!(rhs != 0.0);
        let inv = 1.0 / rhs;

        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    fn assert_vector3i_equal(v1: Vector3i, v2: Vector3i) {
        assert_eq!(v1.x, v2.x);
        assert_eq!(v1.y, v2.y);
        assert_eq!(v1.z, v2.z);
    }

    fn assert_vector3f_equal(v1: Vector3f, v2: Vector3f) {
        assert_approx_eq!(v1.x, v2.x);
        assert_approx_eq!(v1.y, v2.y);
        assert_approx_eq!(v1.z, v2.z);
    }

    // Construction

    #[test]
    fn vector3i_new() {
        let given = Vector3i::new(0, 1, 2);
        let expected = Vector3i { x: 0, y: 1, z: 2 };
        assert_vector3i_equal(given, expected);
    }

    #[test]
    fn vector3f_new() {
        let given = Vector3f::new(0.0, 1.0, 2.0);
        let expected = Vector3f {
            x: 0.0,
            y: 1.0,
            z: 2.0,
        };
        assert_vector3f_equal(given, expected);
    }

    // Methods

    #[test]
    fn vector3_abs() {
        let given = Vector3i::new(-1, 1, -1).abs();
        let expected = Vector3i::new(1, 1, 1).abs();
        assert_vector3i_equal(given, expected);
    }

    #[test]
    fn vector3f_has_nans() {
        let given = Vector3f::new(0.0, 1.0, 2.0);
        assert!(!given.has_nans());
        let given = Vector3f::new(std::f32::NAN, 1.0, 2.0);
        assert!(given.has_nans());
    }

    #[test]
    fn vector3_length_squared() {
        let given = Vector3f::new(2.0, 2.0, 2.0).length_squared();
        let expected = 12.0;
        assert_approx_eq!(given, expected);
    }

    #[test]
    fn vector3_length() {
        let given = Vector3f::new(2.0, 2.0, 2.0).length();
        let expected = 12.0_f32.sqrt();
        assert_approx_eq!(given, expected);
    }

    #[test]
    fn vector3_normalise() {
        let given = Vector3f::new(0.0, 0.0, 1.0).normalise();
        let expected = Vector3f::new(0.0, 0.0, 1.0);
        assert_vector3f_equal(given, expected);
    }

    #[test]
    fn vector3_min_component() {
        let given = Vector3f::new(1.0, 2.0, 3.0).min_component();
        let expected = 1.0;
        assert_approx_eq!(given, expected);
        let given = Vector3i::new(3, 2, 1).min_component();
        let expected = 1;
        assert_eq!(given, expected);
        let given = Vector3i::new(3, 1, 2).min_component();
        let expected = 1;
        assert_eq!(given, expected);
    }

    #[test]
    fn vector3_max_component() {
        let given = Vector3f::new(1.0, 2.0, 3.0).max_component();
        let expected = 3.0;
        assert_approx_eq!(given, expected);
        let given = Vector3i::new(3, 2, 1).max_component();
        let expected = 3;
        assert_eq!(given, expected);
        let given = Vector3i::new(2, 3, 1).max_component();
        let expected = 3;
        assert_eq!(given, expected);
    }

    #[test]
    fn vector3_min_dimension() {
        let given = Vector3f::new(1.0, 2.0, 3.0).min_dimension();
        let expected = 0;
        assert_eq!(given, expected);
        let given = Vector3i::new(3, 2, 1).min_dimension();
        let expected = 2;
        assert_eq!(given, expected);
        let given = Vector3i::new(3, 1, 2).min_dimension();
        let expected = 1;
        assert_eq!(given, expected);
    }

    #[test]
    fn vector3_max_dimension() {
        let given = Vector3f::new(1.0, 2.0, 3.0).max_dimension();
        let expected = 2;
        assert_eq!(given, expected);
        let given = Vector3i::new(3, 2, 1).max_dimension();
        let expected = 0;
        assert_eq!(given, expected);
        let given = Vector3i::new(2, 3, 1).max_dimension();
        let expected = 1;
        assert_eq!(given, expected);
    }

    #[test]
    fn vector3_permute() {
        let given = Vector3i::new(1, 2, 3).permute(2, 1, 0);
        let expected = Vector3i::new(3, 2, 1);
        assert_vector3i_equal(given, expected);
    }

    // Associated functions

    #[test]
    fn vector3_dot() {
        let given = Vector3::dot(Vector3i::new(0, 1, 2), Vector3i::new(1, 2, 3));
        let expected = 8;
        assert_eq!(given, expected);
    }

    #[test]
    fn vector3_dot_abs() {
        let given = Vector3::dot_abs(Vector3i::new(-1, -1, -1), Vector3i::new(1, 2, 3));
        let expected = 6;
        assert_eq!(given, expected);
    }

    #[test]
    fn vector3_cross() {
        let given = Vector3::cross(Vector3i::new(3, -2, -2), Vector3i::new(-1, 0, 5));
        let expected = Vector3i::new(-10, -13, -2);
        assert_vector3i_equal(given, expected);
    }

    #[test]
    fn vector3_coordinate_system() {
        let (given1, given2, given3) = Vector3f::coordinate_system(Vector3f::new(0.0, 0.0, 1.0));
        let (expec1, expec2, expec3) = (
            Vector3f::new(0.0, 0.0, 1.0),
            Vector3f::new(0.0, 1.0, 0.0),
            Vector3f::new(-1.0, 0.0, 0.0),
        );
        assert_vector3f_equal(given1, expec1);
        assert_vector3f_equal(given2, expec2);
        assert_vector3f_equal(given3, expec3);
    }

    // Indexing traits

    #[test]
    fn vector3_index() {
        let given = Vector3i::new(1, 2, 3);
        assert_eq!(given[0], 1);
        assert_eq!(given[1], 2);
        assert_eq!(given[2], 3);
    }

    #[test]
    #[should_panic]
    fn vector3_index_out_of_bounds() {
        let _ = Vector3i::new(1, 2, 3)[3];
    }

    #[test]
    fn vector3_index_mut() {
        let mut given = Vector3i::new(1, 2, 3);
        given[0] = 2;
        given[1] = 3;
        given[2] = 4;
        assert_vector3i_equal(given, Vector3i::new(2, 3, 4));
    }

    #[test]
    #[should_panic]
    fn vector3_index_mut_out_of_bounds() {
        let mut given = Vector3i::new(1, 2, 3);
        given[3] = 4;
    }

    // Negation trait

    #[test]
    fn vector3_neg() {
        let given = -Vector3i::new(-1, 2, -3);
        let expected = Vector3i::new(1, -2, 3);
        assert_vector3i_equal(given, expected)
    }

    // Addition traits

    #[test]
    fn vector3_add() {
        let given = Vector3i::new(0, 1, 2) + Vector3i::new(2, 3, 4);
        let expected = Vector3i::new(2, 4, 6);
        assert_vector3i_equal(given, expected)
    }

    #[test]
    fn vector3_add_assign() {
        let mut given = Vector3i::new(0, 1, 2);
        given += Vector3i::new(2, 3, 4);
        let expected = Vector3i::new(2, 4, 6);
        assert_vector3i_equal(given, expected)
    }

    // Subtraction traits

    #[test]
    fn vector3_sub() {
        let given = Vector3i::new(0, 1, 2) - Vector3i::new(2, 3, 4);
        let expected = Vector3i::new(-2, -2, -2);
        assert_vector3i_equal(given, expected)
    }

    #[test]
    fn vector3_sub_assign() {
        let mut given = Vector3i::new(0, 1, 2);
        given -= Vector3i::new(2, 3, 4);
        let expected = Vector3i::new(-2, -2, -2);
        assert_vector3i_equal(given, expected)
    }

    // Multiplication traits

    #[test]
    fn vector3_mul_vector_scalar() {
        let given = Vector3i::new(0, 1, 2) * 2;
        let expected = Vector3i::new(0, 2, 4);
        assert_vector3i_equal(given, expected)
    }

    #[test]
    fn vector3_mul_scalar_vector() {
        let given = 2 * Vector3i::new(0, 1, 2);
        let expected = Vector3i::new(0, 2, 4);
        assert_vector3i_equal(given, expected);
        let given = 2.0 * Vector3f::new(0.0, 1.0, 2.0);
        let expected = Vector3f::new(0.0, 2.0, 4.0);
        assert_vector3f_equal(given, expected);
    }

    #[test]
    fn vector3_mul_assign() {
        let mut given = Vector3i::new(0, 1, 2);
        given *= 2;
        let expected = Vector3i::new(0, 2, 4);
        assert_vector3i_equal(given, expected);
    }

    // Division traits

    #[test]
    fn vector3_div() {
        let given = Vector3f::new(1.0, 2.0, 3.0) / 2.0;
        let expected = Vector3f::new(0.5, 1.0, 1.5);
        assert_vector3f_equal(given, expected);
    }

    #[test]
    fn vector3_div_assign() {
        let mut given = Vector3f::new(1.0, 2.0, 3.0);
        given /= 2.0;
        let expected = Vector3f::new(0.5, 1.0, 1.5);
        assert_vector3f_equal(given, expected);
    }
}
