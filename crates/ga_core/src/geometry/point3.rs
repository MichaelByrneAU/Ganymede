//! A three-dimensional point.

use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use num::Signed;

use crate::constants::Float;
use crate::geometry::vector3::Vector3;

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
    /// [`Point3`]: struct.Point3.html
    /// [`Point3i`]: type.Point3i.html
    /// [`Point3f`]: type.Point3f.html
    pub fn new(x: T, y: T, z: T) -> Self {
        Point3 { x, y, z }
    }
}

// Methods

impl<T> Point3<T> {
    /// Return a new [`Point3`] with absolute values of its components.
    ///
    /// [`Point3`]: struct.Point3.html
    pub fn abs(self) -> Self
    where
        T: Signed,
    {
        Point3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
}

impl Point3<Float> {
    /// Return a new [`Point3`] with rounded-down values of its components.
    ///
    /// [`Point3`]: struct.Point3.html
    pub fn floor(self) -> Self {
        Point3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Return a new [`Point3`] with rounded-up values of its components.
    ///
    /// [`Point3`]: struct.Point3.html
    pub fn ceil(self) -> Self {
        Point3::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }
}

// Associated functions

impl<T> Point3<T> {
    /// Return the squared distance between two ['Point3']s.
    ///
    /// [`Point3`]: struct.Point3.html
    pub fn distance_squared(p1: Self, p2: Self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
    {
        (p1 - p2).length_squared()
    }

    /// Return the component-wise minimum of two ['Point3']s.
    ///
    /// [`Point3`]: struct.Point3.html
    pub fn min(p1: Self, p2: Self) -> Self
    where
        T: Copy + PartialOrd,
    {
        Point3::new(
            if p1.x < p2.x { p1.x } else { p2.x },
            if p1.y < p2.y { p1.y } else { p2.y },
            if p1.z < p2.z { p1.z } else { p2.z },
        )
    }

    /// Return the component-wise maximum of two ['Point3']s.
    ///
    /// [`Point3`]: struct.Point3.html
    pub fn max(p1: Self, p2: Self) -> Self
    where
        T: Copy + PartialOrd,
    {
        Point3::new(
            if p1.x > p2.x { p1.x } else { p2.x },
            if p1.y > p2.y { p1.y } else { p2.y },
            if p1.z > p2.z { p1.z } else { p2.z },
        )
    }

    /// Return a new [`Point3`] with the coordinates permuted according to the indices
    /// provided.
    ///
    /// [`Point3`]: struct.Point2.html
    pub fn permute(&self, x: usize, y: usize, z: usize) -> Self
    where
        T: Copy,
    {
        Point3::new(self[x], self[y], self[z])
    }
}

impl Point3<Float> {
    /// Return the distance between two ['Point3']s.
    ///
    /// [`Point3`]: struct.Point3.html
    pub fn distance(p1: Self, p2: Self) -> Float {
        (p1 - p2).length()
    }

    /// Linearly interpolate between `p0` an `p1`.
    pub fn lerp(t: Float, p0: Self, p1: Self) -> Self {
        (1.0 - t) * p0 + t * p1
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

// Indexing traits

impl<T> Index<usize> for Point3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("out of bounds access (Point3)"),
        }
    }
}

impl<T> IndexMut<usize> for Point3<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("out of bounds access (Point3)"),
        }
    }
}

// Addition traits

impl<T> Add for Point3<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl<T> AddAssign for Point3<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<T> Add<Vector3<T>> for Point3<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Vector3<T>) -> Self {
        Point3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl<T> AddAssign<Vector3<T>> for Point3<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, other: Vector3<T>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

// Subtraction traits

impl<T> Sub for Point3<T>
where
    T: Sub<Output = T>,
{
    type Output = Vector3<T>;

    fn sub(self, other: Self) -> Self::Output {
        Vector3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T> Sub<Vector3<T>> for Point3<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Vector3<T>) -> Self {
        Point3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T> SubAssign<Vector3<T>> for Point3<T>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, other: Vector3<T>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

// Multiplication traits

impl<T> Mul<T> for Point3<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Point3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Point3i> for i32 {
    type Output = Point3i;

    fn mul(self, rhs: Point3i) -> Point3i {
        Point3i::new(rhs.x * self, rhs.y * self, rhs.z * self)
    }
}

impl Mul<Point3f> for Float {
    type Output = Point3f;

    fn mul(self, rhs: Point3f) -> Point3f {
        Point3f::new(rhs.x * self, rhs.y * self, rhs.z * self)
    }
}

impl<T> MulAssign<T> for Point3<T>
where
    T: Copy + MulAssign<T>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::vector3::Vector3i;

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

    fn assert_vector3i_equal(v1: Vector3i, v2: Vector3i) {
        assert_eq!(v1.x, v2.x);
        assert_eq!(v1.y, v2.y);
        assert_eq!(v1.z, v2.z);
    }

    // Construction

    #[test]
    fn point3i_new() {
        let given = Point3i::new(0, 1, 2);
        let expected = Point3i { x: 0, y: 1, z: 2 };
        assert_point3i_equal(given, expected);
    }

    // Methods

    #[test]
    fn point3_abs() {
        let given = Point3i::new(-1, 1, -1).abs();
        let expected = Point3i::new(1, 1, 1);
        assert_point3i_equal(given, expected);
    }

    #[test]
    fn point3_floor() {
        let given = Point3f::new(-1.5, 1.5, -1.3).floor();
        let expected = Point3f::new(-2.0, 1.0, -2.0);
        assert_point3f_equal(given, expected);
    }

    #[test]
    fn point3_ceil() {
        let given = Point3f::new(-1.5, 1.5, -1.5).ceil();
        let expected = Point3f::new(-1.0, 2.0, -1.0);
        assert_point3f_equal(given, expected);
    }

    // Associated functions

    #[test]
    fn point3_distance_squared() {
        let given = Point3::distance_squared(Point3i::new(0, 0, 0), Point3i::new(1, 1, 1));
        let expected = 3;
        assert_eq!(given, expected);
    }

    #[test]
    fn point3_distance() {
        let given = Point3::distance(Point3f::new(0.0, 0.0, 0.0), Point3f::new(0.0, 1.0, 0.0));
        let expected = 1.0;
        assert_approx_eq!(given, expected);
    }

    #[test]
    fn point3_lerp() {
        let given = Point3::lerp(
            0.5,
            Point3f::new(0.0, 0.0, 0.0),
            Point3f::new(1.0, 1.0, 1.0),
        );
        let expected = Point3f::new(0.5, 0.5, 0.5);
        assert_point3f_equal(given, expected);
    }

    #[test]
    fn point3_min() {
        let given = Point3::min(Point3i::new(0, 3, 2), Point3i::new(1, 2, 2));
        let expected = Point3::new(0, 2, 2);
        assert_point3i_equal(given, expected);
    }

    #[test]
    fn point3_max() {
        let given = Point3::max(Point3i::new(0, 3, 2), Point3i::new(1, 2, 2));
        let expected = Point3::new(1, 3, 2);
        assert_point3i_equal(given, expected);
    }

    #[test]
    fn point2_permute() {
        let given = Point3i::new(1, 2, 3).permute(2, 1, 0);
        let expected = Point3i::new(3, 2, 1);
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

    // Indexing traits

    #[test]
    fn point3_index() {
        let given = Point3i::new(1, 2, 3);
        assert_eq!(given[0], 1);
        assert_eq!(given[1], 2);
        assert_eq!(given[2], 3);
    }

    #[test]
    #[should_panic]
    fn point3_index_out_of_bounds() {
        let _ = Point3i::new(1, 2, 3)[3];
    }

    #[test]
    fn point3_index_mut() {
        let mut given = Point3i::new(1, 2, 3);
        given[0] = 2;
        given[1] = 3;
        given[2] = 4;
        assert_point3i_equal(given, Point3i::new(2, 3, 4));
    }

    #[test]
    #[should_panic]
    fn point3_index_mut_out_of_bounds() {
        let mut given = Point3i::new(1, 2, 3);
        given[3] = 4;
    }

    // Addition traits

    #[test]
    fn point3_add() {
        let given = Point3i::new(0, 1, 2) + Point3i::new(2, 3, 4);
        let expected = Point3i::new(2, 4, 6);
        assert_point3i_equal(given, expected)
    }

    #[test]
    fn point3_add_assign() {
        let mut given = Point3i::new(0, 1, 2);
        given += Point3i::new(2, 3, 4);
        let expected = Point3i::new(2, 4, 6);
        assert_point3i_equal(given, expected)
    }

    #[test]
    fn point3_add_vector3() {
        let given = Point3i::new(0, 0, 0) + Vector3i::new(1, 2, 3);
        let expected = Point3i::new(1, 2, 3);
        assert_point3i_equal(given, expected);
    }

    #[test]
    fn point2_add_assign_vector3() {
        let mut given = Point3i::new(0, 0, 0);
        given += Vector3i::new(1, 2, 3);
        let expected = Point3i::new(1, 2, 3);
        assert_point3i_equal(given, expected);
    }

    // Subtraction traits

    #[test]
    fn point3_sub_point3() {
        let given = Point3i::new(2, 1, 0) - Point3i::new(2, 2, 2);
        let expected = Vector3i::new(0, -1, -2);
        assert_vector3i_equal(given, expected);
    }

    #[test]
    fn point3_sub_vector3() {
        let given = Point3i::new(2, 1, 0) - Vector3i::new(2, 2, 2);
        let expected = Point3i::new(0, -1, -2);
        assert_point3i_equal(given, expected);
    }

    #[test]
    fn point3_sub_assign_vector3() {
        let mut given = Point3i::new(2, 1, 0);
        given -= Vector3i::new(2, 2, 2);
        let expected = Point3i::new(0, -1, -2);
        assert_point3i_equal(given, expected);
    }

    // Multiplication traits

    #[test]
    fn point3_mul_point_scalar() {
        let given = Point3i::new(0, 1, 2) * 2;
        let expected = Point3i::new(0, 2, 4);
        assert_point3i_equal(given, expected)
    }

    #[test]
    fn point3_mul_scalar_point() {
        let given = 2 * Point3i::new(0, 1, 2);
        let expected = Point3i::new(0, 2, 4);
        assert_point3i_equal(given, expected);
        let given = 2.0 * Point3f::new(0.0, 1.0, 2.0);
        let expected = Point3f::new(0.0, 2.0, 4.0);
        assert_point3f_equal(given, expected);
    }

    #[test]
    fn point3_mul_assign() {
        let mut given = Point3i::new(0, 1, 2);
        given *= 2;
        let expected = Point3i::new(0, 2, 4);
        assert_point3i_equal(given, expected);
    }
}
