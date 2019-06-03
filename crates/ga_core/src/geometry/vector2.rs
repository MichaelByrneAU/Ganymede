//! A two-dimensional vector.

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use num::Signed;

use crate::constants::Float;
use crate::geometry::point2::{Point2, Point2f, Point2i};

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

// Methods

impl<T> Vector2<T> {
    /// Return a new [`Vector2`] with absolute values of its components.
    ///
    /// [`Vector2`]: struct.Vector2.html
    pub fn abs(self) -> Self
    where
        T: Signed,
    {
        Vector2::new(self.x.abs(), self.y.abs())
    }

    /// Return the squared length of the [`Vector2`].
    ///
    /// [`Vector2`]: struct.Vector2.html
    pub fn length_squared(self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        self.x * self.x + self.y * self.y
    }

    /// Return the minimum component of the [`Vector2`].
    ///
    /// [`Vector2`]: struct.Vector2.html
    pub fn min_component(self) -> T
    where
        T: Copy + PartialOrd,
    {
        if self.x > self.y {
            self.y
        } else {
            self.x
        }
    }

    /// Return the maximum component of the [`Vector2`].
    ///
    /// [`Vector2`]: struct.Vector2.html
    pub fn max_component(self) -> T
    where
        T: Copy + PartialOrd,
    {
        if self.x < self.y {
            self.y
        } else {
            self.x
        }
    }

    /// Return the index corresponding to the minimum component of the [`Vector2`].
    ///
    /// [`Vector2`]: struct.Vector2.html
    pub fn min_dimension(self) -> usize
    where
        T: Copy + PartialOrd,
    {
        if self.x > self.y {
            1
        } else {
            0
        }
    }

    /// Return the index corresponding to the maximum component of the [`Vector2`].
    ///
    /// [`Vector2`]: struct.Vector2.html
    pub fn max_dimension(self) -> usize
    where
        T: Copy + PartialOrd,
    {
        if self.x < self.y {
            1
        } else {
            0
        }
    }

    /// Return a new [`Vector2`] with the coordinates permuted according to the indices
    /// provided.
    ///
    /// [`Vector2`]: struct.Vector2.html
    pub fn permute(self, x: usize, y: usize) -> Self
    where
        T: Copy,
    {
        Vector2::new(self[x], self[y])
    }
}

impl Vector2<Float> {
    /// Check whether any component holds a NaN value.
    pub fn has_nans(self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }

    /// Return the length of the [`Vector2`].
    ///
    /// [`Vector2`]: struct.Vector2.html
    pub fn length(self) -> Float {
        self.length_squared().sqrt()
    }

    /// Return a normalised version of a [`Vector2`].
    ///
    /// [`Vector2`]: struct.Vector2.html
    pub fn normalise(self) -> Self {
        self / self.length()
    }
}

// Associated functions

impl<T> Vector2<T> {
    /// Compute the dot product of two [`Vector2`]s.
    ///
    /// # Examples
    /// ```
    /// # use ga_core::geometry::vector2::{Vector2, Vector2i};
    /// let v1 = Vector2i::new(0, 1);
    /// let v2 = Vector2i::new(1, 2);
    /// assert_eq!(Vector2::dot(v1, v2), 2);
    /// ```
    ///
    /// [`Vector2`]: struct.Vector2.html
    pub fn dot(v1: Self, v2: Self) -> T
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        v1.x * v2.x + v1.y * v2.y
    }

    /// Compute the absolute dot product of two [`Vector2`]s.
    ///
    /// # Examples
    /// ```
    /// # use ga_core::geometry::vector2::{Vector2, Vector2i};
    /// let v1 = Vector2i::new(-1, -1);
    /// let v2 = Vector2i::new(1, 2);
    /// assert_eq!(Vector2::dot_abs(v1, v2), 3);
    /// ```
    ///
    /// [`Vector2`]: struct.Vector2.html
    pub fn dot_abs(v1: Self, v2: Self) -> T
    where
        T: Signed + Add<Output = T> + Mul<Output = T>,
    {
        Vector2::dot(v1, v2).abs()
    }
}

// From traits

impl<T> From<Point2<T>> for Vector2<T> {
    fn from(p2: Point2<T>) -> Self {
        Vector2::new(p2.x, p2.y)
    }
}

impl From<Point2i> for Vector2f {
    fn from(p2i: Point2i) -> Self {
        Vector2f::new(p2i.x as Float, p2i.y as Float)
    }
}

impl From<Point2f> for Vector2i {
    fn from(p2f: Point2f) -> Self {
        Vector2i::new(p2f.x as i32, p2f.y as i32)
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

// Negation trait

impl<T> Neg for Vector2<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self {
        Vector2::new(-self.x, -self.y)
    }
}

// Addition traits

impl<T> Add for Vector2<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Vector2::new(self.x + other.x, self.y + other.y)
    }
}

impl<T> AddAssign for Vector2<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

// Subtraction traits

impl<T> Sub for Vector2<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Vector2::new(self.x - other.x, self.y - other.y)
    }
}

impl<T> SubAssign for Vector2<T>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

// Multiplication traits

impl<T> Mul<T> for Vector2<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Vector2::new(self.x * rhs, self.y * rhs)
    }
}

impl Mul<Vector2i> for i32 {
    type Output = Vector2i;

    fn mul(self, rhs: Vector2i) -> Vector2i {
        Vector2i::new(rhs.x * self, rhs.y * self)
    }
}

impl Mul<Vector2f> for Float {
    type Output = Vector2f;

    fn mul(self, rhs: Vector2f) -> Vector2f {
        Vector2f::new(rhs.x * self, rhs.y * self)
    }
}

impl<T> MulAssign<T> for Vector2<T>
where
    T: Copy + MulAssign<T>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

// Division traits

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<Float> for Vector2f {
    type Output = Self;

    fn div(self, rhs: Float) -> Self {
        debug_assert!(rhs != 0.0);
        let inv = 1.0 / rhs;

        Vector2f::new(self.x * inv, self.y * inv)
    }
}

impl DivAssign<Float> for Vector2f {
    fn div_assign(&mut self, rhs: Float) {
        debug_assert!(rhs != 0.0);
        let inv = 1.0 / rhs;

        self.x *= inv;
        self.y *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point2::{Point2f, Point2i};

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

    // Methods

    #[test]
    fn vector2_abs() {
        let given = Vector2i::new(-1, 1).abs();
        let expected = Vector2i::new(1, 1);
        assert_vector2i_equal(given, expected);
    }

    #[test]
    fn vector2f_has_nans() {
        let given = Vector2f::new(0.0, 1.0);
        assert!(!given.has_nans());
        let given = Vector2f::new(std::f32::NAN, 1.0);
        assert!(given.has_nans());
    }

    #[test]
    fn vector2_length_squared() {
        let given = Vector2f::new(2.0, 2.0).length_squared();
        let expected = 8.0;
        assert_approx_eq!(given, expected);
    }

    #[test]
    fn vector2_length() {
        let given = Vector2f::new(2.0, 2.0).length();
        let expected = 8.0_f32.sqrt();
        assert_approx_eq!(given, expected);
    }

    #[test]
    fn vector2_normalise() {
        let given = Vector2f::new(0.0, 1.0).normalise();
        let expected = Vector2f::new(0.0, 1.0);
        assert_vector2f_equal(given, expected);
    }

    #[test]
    fn vector2_min_component() {
        let given = Vector2f::new(1.0, 2.0).min_component();
        let expected = 1.0;
        assert_approx_eq!(given, expected);
        let given = Vector2i::new(2, 1).min_component();
        let expected = 1;
        assert_eq!(given, expected);
    }

    #[test]
    fn vector2_max_component() {
        let given = Vector2f::new(1.0, 2.0).max_component();
        let expected = 2.0;
        assert_approx_eq!(given, expected);
        let given = Vector2i::new(2, 1).max_component();
        let expected = 2;
        assert_eq!(given, expected);
    }

    #[test]
    fn vector2_min_dimension() {
        let given = Vector2f::new(1.0, 2.0).min_dimension();
        let expected = 0;
        assert_eq!(given, expected);
        let given = Vector2i::new(2, 1).min_dimension();
        let expected = 1;
        assert_eq!(given, expected);
    }

    #[test]
    fn vector2_max_dimension() {
        let given = Vector2f::new(1.0, 2.0).max_dimension();
        let expected = 1;
        assert_eq!(given, expected);
        let given = Vector2i::new(2, 1).max_dimension();
        let expected = 0;
        assert_eq!(given, expected);
    }

    #[test]
    fn vector2_permute() {
        let given = Vector2i::new(1, 2).permute(1, 0);
        let expected = Vector2i::new(2, 1);
        assert_vector2i_equal(given, expected);
    }

    // Associated functions

    #[test]
    fn vector2_dot() {
        let given = Vector2::dot(Vector2i::new(0, 1), Vector2i::new(1, 2));
        let expected = 2;
        assert_eq!(given, expected);
    }

    #[test]
    fn vector2_dot_abs() {
        let given = Vector2::dot_abs(Vector2i::new(-1, -1), Vector2i::new(1, 2));
        let expected = 3;
        assert_eq!(given, expected);
    }

    // From traits

    #[test]
    fn vector2i_from_point2() {
        let given = Vector2i::from(Point2i::new(0, 1));
        let expected = Vector2i::new(0, 1);
        assert_vector2i_equal(given, expected);
        let given = Vector2i::from(Point2f::new(0.0, 1.0));
        let expected = Vector2i::new(0, 1);
        assert_vector2i_equal(given, expected);
    }

    #[test]
    fn vector2f_from_point2() {
        let given = Vector2f::from(Point2i::new(0, 1));
        let expected = Vector2f::new(0.0, 1.0);
        assert_vector2f_equal(given, expected);
        let given = Vector2f::from(Point2f::new(0.0, 1.0));
        let expected = Vector2f::new(0.0, 1.0);
        assert_vector2f_equal(given, expected);
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

    // Negation trait

    #[test]
    fn vector2_neg() {
        let given = -Vector2i::new(-1, 2);
        let expected = Vector2i::new(1, -2);
        assert_vector2i_equal(given, expected)
    }

    // Addition traits

    #[test]
    fn vector2_add() {
        let given = Vector2i::new(0, 1) + Vector2i::new(2, 3);
        let expected = Vector2i::new(2, 4);
        assert_vector2i_equal(given, expected)
    }

    #[test]
    fn vector2_add_assign() {
        let mut given = Vector2i::new(0, 1);
        given += Vector2i::new(2, 3);
        let expected = Vector2i::new(2, 4);
        assert_vector2i_equal(given, expected)
    }

    // Subtraction traits

    #[test]
    fn vector2_sub() {
        let given = Vector2i::new(0, 1) - Vector2i::new(2, 3);
        let expected = Vector2i::new(-2, -2);
        assert_vector2i_equal(given, expected)
    }

    #[test]
    fn vector2_sub_assign() {
        let mut given = Vector2i::new(0, 1);
        given -= Vector2i::new(2, 3);
        let expected = Vector2i::new(-2, -2);
        assert_vector2i_equal(given, expected)
    }

    // Multiplication traits

    #[test]
    fn vector2_mul_vector_scalar() {
        let given = Vector2i::new(0, 1) * 2;
        let expected = Vector2i::new(0, 2);
        assert_vector2i_equal(given, expected)
    }

    #[test]
    fn vector2_mul_scalar_vector() {
        let given = 2 * Vector2i::new(0, 1);
        let expected = Vector2i::new(0, 2);
        assert_vector2i_equal(given, expected);
        let given = 2.0 * Vector2f::new(0.0, 1.0);
        let expected = Vector2f::new(0.0, 2.0);
        assert_vector2f_equal(given, expected);
    }

    #[test]
    fn vector2_mul_assign() {
        let mut given = Vector2i::new(0, 1);
        given *= 2;
        let expected = Vector2i::new(0, 2);
        assert_vector2i_equal(given, expected);
    }

    // Division traits

    #[test]
    fn vector2_div() {
        let given = Vector2f::new(1.0, 2.0) / 2.0;
        let expected = Vector2f::new(0.5, 1.0);
        assert_vector2f_equal(given, expected);
    }

    #[test]
    fn vector2_div_assign() {
        let mut given = Vector2f::new(1.0, 2.0);
        given /= 2.0;
        let expected = Vector2f::new(0.5, 1.0);
        assert_vector2f_equal(given, expected);
    }

}
