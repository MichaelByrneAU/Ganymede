//! A two-dimensional point.

use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use num::Signed;

use crate::constants::Float;
use crate::geometry::point3::Point3;
use crate::geometry::vector2::Vector2;

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
    /// [`Point2`]: struct.Point2.html
    /// [`Point2i`]: type.Point2i.html
    /// [`Point2f`]: type.Point2f.html
    pub fn new(x: T, y: T) -> Self {
        Point2 { x, y }
    }
}

// Methods

impl<T> Point2<T> {
    /// Return a new [`Point2`] with absolute values of its components.
    ///
    /// [`Point2`]: struct.Point2.html
    pub fn abs(self) -> Self
    where
        T: Signed,
    {
        Point2::new(self.x.abs(), self.y.abs())
    }
}

impl Point2<Float> {
    /// Return a new [`Point2`] with rounded-down values of its components.
    ///
    /// [`Point2`]: struct.Point2.html
    pub fn floor(self) -> Self {
        Point2::new(self.x.floor(), self.y.floor())
    }

    /// Return a new [`Point2`] with rounded-up values of its components.
    ///
    /// [`Point2`]: struct.Point2.html
    pub fn ceil(self) -> Self {
        Point2::new(self.x.ceil(), self.y.ceil())
    }
}

// Associated functions

impl<T> Point2<T> {
    /// Return the squared distance between two ['Point2']s.
    ///
    /// [`Point2`]: struct.Point2.html
    pub fn distance_squared(p1: Self, p2: Self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
    {
        (p1 - p2).length_squared()
    }

    /// Return the component-wise minimum of two ['Point2']s.
    ///
    /// [`Point2`]: struct.Point2.html
    pub fn min(p1: Self, p2: Self) -> Self
    where
        T: Copy + PartialOrd,
    {
        Point2::new(
            if p1.x < p2.x { p1.x } else { p2.x },
            if p1.y < p2.y { p1.y } else { p2.y },
        )
    }

    /// Return the component-wise maximum of two ['Point2']s.
    ///
    /// [`Point2`]: struct.Point2.html
    pub fn max(p1: Self, p2: Self) -> Self
    where
        T: Copy + PartialOrd,
    {
        Point2::new(
            if p1.x > p2.x { p1.x } else { p2.x },
            if p1.y > p2.y { p1.y } else { p2.y },
        )
    }

    /// Return a new [`Point2`] with the coordinates permuted according to the indices
    /// provided.
    ///
    /// [`Point2`]: struct.Point2.html
    pub fn permute(&self, x: usize, y: usize) -> Self
    where
        T: Copy,
    {
        Point2::new(self[x], self[y])
    }
}

impl Point2<Float> {
    /// Return the distance between two ['Point2']s.
    ///
    /// [`Point2`]: struct.Point2.html
    pub fn distance(p1: Self, p2: Self) -> Float {
        (p1 - p2).length()
    }

    /// Linearly interpolate between `p0` an `p1`.
    pub fn lerp(t: Float, p0: Self, p1: Self) -> Self {
        (1.0 - t) * p0 + t * p1
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

// Indexing traits

impl<T> Index<usize> for Point2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("out of bounds access (Point2)"),
        }
    }
}

impl<T> IndexMut<usize> for Point2<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("out of bounds access (Point2)"),
        }
    }
}

// Addition traits

impl<T> Add for Point2<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Point2::new(self.x + other.x, self.y + other.y)
    }
}

impl<T> AddAssign for Point2<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl<T> Add<Vector2<T>> for Point2<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Vector2<T>) -> Self {
        Point2::new(self.x + other.x, self.y + other.y)
    }
}

impl<T> AddAssign<Vector2<T>> for Point2<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, other: Vector2<T>) {
        self.x += other.x;
        self.y += other.y;
    }
}

// Subtraction traits

impl<T> Sub for Point2<T>
where
    T: Sub<Output = T>,
{
    type Output = Vector2<T>;

    fn sub(self, other: Self) -> Self::Output {
        Vector2::new(self.x - other.x, self.y - other.y)
    }
}

impl<T> Sub<Vector2<T>> for Point2<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Vector2<T>) -> Self {
        Point2::new(self.x - other.x, self.y - other.y)
    }
}

impl<T> SubAssign<Vector2<T>> for Point2<T>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, other: Vector2<T>) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

// Multiplication traits

impl<T> Mul<T> for Point2<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Point2::new(self.x * rhs, self.y * rhs)
    }
}

impl Mul<Point2i> for i32 {
    type Output = Point2i;

    fn mul(self, rhs: Point2i) -> Point2i {
        Point2i::new(rhs.x * self, rhs.y * self)
    }
}

impl Mul<Point2f> for Float {
    type Output = Point2f;

    fn mul(self, rhs: Point2f) -> Point2f {
        Point2f::new(rhs.x * self, rhs.y * self)
    }
}

impl<T> MulAssign<T> for Point2<T>
where
    T: Copy + MulAssign<T>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point3::Point3i;
    use crate::geometry::vector2::Vector2i;

    use assert_approx_eq::assert_approx_eq;

    fn assert_point2i_equal(p1: Point2i, p2: Point2i) {
        assert_eq!(p1.x, p2.x);
        assert_eq!(p1.y, p2.y);
    }

    fn assert_point2f_equal(p1: Point2f, p2: Point2f) {
        assert_approx_eq!(p1.x, p2.x);
        assert_approx_eq!(p1.y, p2.y);
    }

    fn assert_vector2i_equal(v1: Vector2i, v2: Vector2i) {
        assert_eq!(v1.x, v2.x);
        assert_eq!(v1.y, v2.y);
    }

    // Construction

    #[test]
    fn point2i_new() {
        let given = Point2i::new(0, 1);
        let expected = Point2i { x: 0, y: 1 };
        assert_point2i_equal(given, expected);
    }

    // Methods

    #[test]
    fn point2_abs() {
        let given = Point2i::new(-1, 1).abs();
        let expected = Point2i::new(1, 1);
        assert_point2i_equal(given, expected);
    }

    #[test]
    fn point2_floor() {
        let given = Point2f::new(-1.5, 1.5).floor();
        let expected = Point2f::new(-2.0, 1.0);
        assert_point2f_equal(given, expected);
    }

    #[test]
    fn point2_ceil() {
        let given = Point2f::new(-1.5, 1.5).ceil();
        let expected = Point2f::new(-1.0, 2.0);
        assert_point2f_equal(given, expected);
    }

    // Associated functions

    #[test]
    fn point2_distance_squared() {
        let given = Point2::distance_squared(Point2i::new(0, 0), Point2i::new(1, 1));
        let expected = 2;
        assert_eq!(given, expected);
    }

    #[test]
    fn point2_distance() {
        let given = Point2::distance(Point2f::new(0.0, 0.0), Point2f::new(0.0, 1.0));
        let expected = 1.0;
        assert_approx_eq!(given, expected);
    }

    #[test]
    fn point2_lerp() {
        let given = Point2::lerp(0.5, Point2f::new(0.0, 0.0), Point2f::new(1.0, 1.0));
        let expected = Point2f::new(0.5, 0.5);
        assert_point2f_equal(given, expected);
    }

    #[test]
    fn point2_min() {
        let given = Point2::min(Point2i::new(0, 3), Point2i::new(1, 2));
        let expected = Point2::new(0, 2);
        assert_point2i_equal(given, expected);
    }

    #[test]
    fn point2_max() {
        let given = Point2::max(Point2i::new(0, 3), Point2i::new(1, 2));
        let expected = Point2::new(1, 3);
        assert_point2i_equal(given, expected);
    }

    #[test]
    fn point2_permute() {
        let given = Point2i::new(1, 2).permute(1, 0);
        let expected = Point2i::new(2, 1);
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

    // Indexing traits

    #[test]
    fn point2_index() {
        let given = Point2i::new(1, 2);
        assert_eq!(given[0], 1);
        assert_eq!(given[1], 2);
    }

    #[test]
    #[should_panic]
    fn point2_index_out_of_bounds() {
        let _ = Point2i::new(1, 2)[2];
    }

    // Addition traits

    #[test]
    fn point2_add() {
        let given = Point2i::new(0, 1) + Point2i::new(2, 3);
        let expected = Point2i::new(2, 4);
        assert_point2i_equal(given, expected)
    }

    #[test]
    fn point2_add_assign() {
        let mut given = Point2i::new(0, 1);
        given += Point2i::new(2, 3);
        let expected = Point2i::new(2, 4);
        assert_point2i_equal(given, expected)
    }

    #[test]
    fn point2_add_vector2() {
        let given = Point2i::new(0, 0) + Vector2i::new(1, 2);
        let expected = Point2i::new(1, 2);
        assert_point2i_equal(given, expected);
    }

    #[test]
    fn point2_add_assign_vector2() {
        let mut given = Point2i::new(0, 0);
        given += Vector2i::new(1, 2);
        let expected = Point2i::new(1, 2);
        assert_point2i_equal(given, expected);
    }

    // Subtraction traits

    #[test]
    fn point2_sub_point2() {
        let given = Point2i::new(2, 1) - Point2i::new(2, 2);
        let expected = Vector2i::new(0, -1);
        assert_vector2i_equal(given, expected);
    }

    #[test]
    fn point2_sub_vector2() {
        let given = Point2i::new(2, 1) - Vector2i::new(2, 2);
        let expected = Point2i::new(0, -1);
        assert_point2i_equal(given, expected);
    }

    #[test]
    fn point2_sub_assign_vector2() {
        let mut given = Point2i::new(2, 1);
        given -= Vector2i::new(2, 2);
        let expected = Point2i::new(0, -1);
        assert_point2i_equal(given, expected);
    }

    // Multiplication traits

    #[test]
    fn point2_mul_point_scalar() {
        let given = Point2i::new(0, 1) * 2;
        let expected = Point2i::new(0, 2);
        assert_point2i_equal(given, expected)
    }

    #[test]
    fn point2_mul_scalar_point() {
        let given = 2 * Point2i::new(0, 1);
        let expected = Point2i::new(0, 2);
        assert_point2i_equal(given, expected);
        let given = 2.0 * Point2f::new(0.0, 1.0);
        let expected = Point2f::new(0.0, 2.0);
        assert_point2f_equal(given, expected);
    }

    #[test]
    fn point2_mul_assign() {
        let mut given = Point2i::new(0, 1);
        given *= 2;
        let expected = Point2i::new(0, 2);
        assert_point2i_equal(given, expected);
    }
}
