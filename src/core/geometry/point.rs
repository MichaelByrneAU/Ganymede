//! Point representations for two and three dimensions.

use std::cmp::{max, min};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use noisy_float::prelude::*;
use num::{abs, Float, Signed};

use super::ops::Length;
use super::vector::{Vector2, Vector3};
use super::{Prim, PrimFloat, PrimItem, PrimSigned};

/// Two-dimensional point of integers.
pub type Point2i = Point2<isize>;
/// Two-dimensional point of floats.
pub type Point2f = Point2<R64>;
/// Three-dimensional point of integers.
pub type Point3i = Point3<isize>;
/// Three-dimensional point of floats.
pub type Point3f = Point3<R64>;

/// A generic, two-dimensional point.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct Point2<T: PrimItem> {
    x: T,
    y: T,
}

// Prim trait implementation

impl<T: PrimItem> Prim for Point2<T> {
    type Item = T;

    fn min_component(self) -> Self::Item {
        min(self.x, self.y)
    }

    fn max_component(self) -> Self::Item {
        max(self.x, self.y)
    }

    fn min_dimension(self) -> usize {
        if self.x < self.y {
            0
        } else {
            1
        }
    }

    fn max_dimension(self) -> usize {
        if self.x > self.y {
            0
        } else {
            1
        }
    }

    fn min(self, other: Self) -> Self {
        Point2 {
            x: min(self.x, other.x),
            y: min(self.y, other.y),
        }
    }

    fn max(self, other: Self) -> Self {
        Point2 {
            x: max(self.x, other.x),
            y: max(self.y, other.y),
        }
    }
}

// PrimSigned trait implementation

impl<T: PrimItem + Signed> PrimSigned for Point2<T> {
    fn abs(self) -> Self {
        Point2 {
            x: abs(self.x),
            y: abs(self.y),
        }
    }
}

// PrimFloat trait implementation

impl<T: PrimItem + Float> PrimFloat for Point2<T> {
    fn floor(self) -> Self {
        Point2 {
            x: self.x.floor(),
            y: self.y.floor(),
        }
    }

    fn ceil(self) -> Self {
        Point2 {
            x: self.x.ceil(),
            y: self.y.ceil(),
        }
    }
}

// Constructor

impl<T: PrimItem> Point2<T> {
    /// Construct a new Point2 from individual component values.
    pub fn new(x: T, y: T) -> Self {
        Point2 { x, y }
    }
}

// Indexing traits

impl<T: PrimItem> Index<usize> for Point2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index out of bounds (Point2)"),
        }
    }
}

impl<T: PrimItem> IndexMut<usize> for Point2<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index out of bounds (Point2)"),
        }
    }
}

impl<T: PrimItem> Point2<T> {
    pub fn permute(&self, x: usize, y: usize) -> Point2<T> {
        Point2 {
            x: self[x],
            y: self[y],
        }
    }
}

// Operator traits

impl<T: PrimItem + Add<Output = T>> Add for Point2<T> {
    type Output = Point2<T>;

    fn add(self, other: Point2<T>) -> Self::Output {
        Point2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<T: PrimItem + Add<Output = T>> Add<Vector2<T>> for Point2<T> {
    type Output = Point2<T>;

    fn add(self, other: Vector2<T>) -> Self::Output {
        Point2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<T: PrimItem + AddAssign> AddAssign for Point2<T> {
    fn add_assign(&mut self, other: Point2<T>) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl<T: PrimItem + Sub<Output = T>> Sub for Point2<T> {
    type Output = Vector2<T>;

    fn sub(self, other: Point2<T>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<T: PrimItem + Sub<Output = T>> Sub<Vector2<T>> for Point2<T> {
    type Output = Point2<T>;

    fn sub(self, other: Vector2<T>) -> Self::Output {
        Point2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<T: PrimItem + SubAssign> SubAssign<Vector2<T>> for Point2<T> {
    fn sub_assign(&mut self, other: Vector2<T>) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl<T: PrimItem + Mul<T, Output = T>> Mul<T> for Point2<T> {
    type Output = Point2<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Point2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T: PrimItem + MulAssign> MulAssign<T> for Point2<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl<T: PrimItem + Div<T, Output = T>> Div<T> for Point2<T> {
    type Output = Point2<T>;

    fn div(self, rhs: T) -> Self::Output {
        Point2 {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl<T: PrimItem + DivAssign> DivAssign<T> for Point2<T> {
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl<T: PrimItem + Signed> Neg for Point2<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Point2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

// Distance operation

impl<T: PrimItem + Float> Point2<T> {
    /// Compute the squared distance between two points.
    pub fn distance_squared(self, rhs: Self) -> T {
        (self - rhs).length_squared()
    }

    /// Compute the distance between two points.
    pub fn distance(self, rhs: Self) -> T {
        (self - rhs).length()
    }
}

// Linear interpolation function
pub fn lerp2(t: R64, p0: Point2f, p1: Point2f) -> Point2f {
    p0 * (r64(1.0) - t) + p1 * t
}

/// A generic, three-dimensional point.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct Point3<T: PrimItem> {
    x: T,
    y: T,
    z: T,
}

// Prim trait implementation

impl<T: PrimItem> Prim for Point3<T> {
    type Item = T;

    fn min_component(self) -> Self::Item {
        min(self.x, min(self.y, self.z))
    }

    fn max_component(self) -> Self::Item {
        max(self.x, max(self.y, self.z))
    }

    fn min_dimension(self) -> usize {
        if self.x < self.y && self.x < self.z {
            0
        } else if self.y < self.z {
            1
        } else {
            2
        }
    }

    fn max_dimension(self) -> usize {
        if self.x > self.y && self.x > self.z {
            0
        } else if self.y > self.z {
            1
        } else {
            2
        }
    }

    fn min(self, other: Self) -> Self {
        Point3 {
            x: min(self.x, other.x),
            y: min(self.y, other.y),
            z: min(self.z, other.z),
        }
    }

    fn max(self, other: Self) -> Self {
        Point3 {
            x: max(self.x, other.x),
            y: max(self.y, other.y),
            z: max(self.z, other.z),
        }
    }
}

// PrimSigned trait implementation

impl<T: PrimItem + Signed> PrimSigned for Point3<T> {
    fn abs(self) -> Self {
        Point3 {
            x: abs(self.x),
            y: abs(self.y),
            z: abs(self.z),
        }
    }
}

// PrimFloat trait implementation

impl<T: PrimItem + Float> PrimFloat for Point3<T> {
    fn floor(self) -> Self {
        Point3 {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
        }
    }

    fn ceil(self) -> Self {
        Point3 {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
        }
    }
}

// Constructors

impl<T: PrimItem> Point3<T> {
    /// Construct a new Point3 from individual component values.
    pub fn new(x: T, y: T, z: T) -> Self {
        Point3 { x, y, z }
    }

    /// Construct a new Point2 from a Point3.
    pub fn to_point2(self) -> Point2<T> {
        Point2 {
            x: self.x,
            y: self.y,
        }
    }
}

// Indexing traits

impl<T: PrimItem> Index<usize> for Point3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index out of bounds (Point3)"),
        }
    }
}

impl<T: PrimItem> IndexMut<usize> for Point3<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index out of bounds (Point3)"),
        }
    }
}

impl<T: PrimItem> Point3<T> {
    pub fn permute(&self, x: usize, y: usize, z: usize) -> Point3<T> {
        Point3 {
            x: self[x],
            y: self[y],
            z: self[z],
        }
    }
}

// Operator traits

impl<T: PrimItem + Add<Output = T>> Add for Point3<T> {
    type Output = Point3<T>;

    fn add(self, other: Point3<T>) -> Self::Output {
        Point3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: PrimItem + Add<Output = T>> Add<Vector3<T>> for Point3<T> {
    type Output = Point3<T>;

    fn add(self, other: Vector3<T>) -> Self::Output {
        Point3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: PrimItem + AddAssign> AddAssign for Point3<T> {
    fn add_assign(&mut self, other: Point3<T>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<T: PrimItem + Sub<Output = T>> Sub for Point3<T> {
    type Output = Vector3<T>;

    fn sub(self, other: Point3<T>) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: PrimItem + Sub<Output = T>> Sub<Vector3<T>> for Point3<T> {
    type Output = Point3<T>;

    fn sub(self, other: Vector3<T>) -> Self::Output {
        Point3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: PrimItem + SubAssign> SubAssign<Vector3<T>> for Point3<T> {
    fn sub_assign(&mut self, other: Vector3<T>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<T: PrimItem + Mul<T, Output = T>> Mul<T> for Point3<T> {
    type Output = Point3<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Point3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T: PrimItem + MulAssign> MulAssign<T> for Point3<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl<T: PrimItem + Div<T, Output = T>> Div<T> for Point3<T> {
    type Output = Point3<T>;

    fn div(self, rhs: T) -> Self::Output {
        Point3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T: PrimItem + DivAssign> DivAssign<T> for Point3<T> {
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl<T: PrimItem + Signed> Neg for Point3<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Point3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// Distance operation

impl<T: PrimItem + Float> Point3<T> {
    /// Compute the squared distance between two points.
    pub fn distance_squared(self, rhs: Self) -> T {
        (self - rhs).length_squared()
    }

    /// Compute the distance between two points.
    pub fn distance(self, rhs: Self) -> T {
        (self - rhs).length()
    }
}

// Linear interpolation function
pub fn lerp3(t: R64, p0: Point3<R64>, p1: Point3<R64>) -> Point3<R64> {
    p0 * (r64(1.0) - t) + p1 * t
}

#[cfg(test)]
#[cfg_attr(tarpaulin, skip)]
mod tests {
    use super::super::vector::{Vector2f, Vector2i, Vector3f, Vector3i};
    use super::*;

    // Point2 tests

    #[test]
    fn point2_min_component() {
        let p1 = Point2i::new(1, 2);
        let p2 = Point2f::new(r64(1.0), r64(2.0));

        assert_eq!(p1.min_component(), 1);
        assert_eq!(p2.min_component(), r64(1.0));
    }

    #[test]
    fn point2_max_component() {
        let p1 = Point2i::new(1, 2);
        let p2 = Point2f::new(r64(1.0), r64(2.0));

        assert_eq!(p1.max_component(), 2);
        assert_eq!(p2.max_component(), r64(2.0));
    }

    #[test]
    fn point2_min_dimension() {
        let p1 = Point2i::new(1, 2);
        let p2 = Point2f::new(r64(1.0), r64(2.0));

        assert_eq!(p1.min_dimension(), 0);
        assert_eq!(p2.min_dimension(), 0);
    }

    #[test]
    fn point2_max_dimension() {
        let p1 = Point2i::new(1, 2);
        let p2 = Point2f::new(r64(1.0), r64(2.0));

        assert_eq!(p1.max_dimension(), 1);
        assert_eq!(p2.max_dimension(), 1);
    }

    #[test]
    fn point2_min() {
        let p1 = Point2i::new(2, 8);
        let p2 = Point2i::new(3, 5);
        let p3 = Point2f::new(r64(2.0), r64(8.0));
        let p4 = Point2f::new(r64(3.0), r64(5.0));

        assert_eq!(p1.min(p2), Point2i::new(2, 5));
        assert_eq!(p3.min(p4), Point2f::new(r64(2.0), r64(5.0)));
    }

    #[test]
    fn point2_max() {
        let p1 = Point2i::new(2, 8);
        let p2 = Point2i::new(3, 5);
        let p3 = Point2f::new(r64(2.0), r64(8.0));
        let p4 = Point2f::new(r64(3.0), r64(5.0));

        assert_eq!(p1.max(p2), Point2i::new(3, 8));
        assert_eq!(p3.max(p4), Point2f::new(r64(3.0), r64(8.0)));
    }

    #[test]
    fn point2_abs() {
        let p1 = Point2i::new(-1, 2);
        let p2 = Point2f::new(r64(-1.0), r64(2.0));

        assert_eq!(p1.abs(), Point2i::new(1, 2));
        assert_eq!(p2.abs(), Point2f::new(r64(1.0), r64(2.0)));
    }

    #[test]
    fn point2_floor() {
        let p1 = Point2f::new(r64(1.1), r64(4.2));

        assert_eq!(p1.floor(), Point2f::new(r64(1.0), r64(4.0)));
    }

    #[test]
    fn point2_ceil() {
        let p1 = Point2f::new(r64(1.1), r64(4.2));

        assert_eq!(p1.ceil(), Point2f::new(r64(2.0), r64(5.0)));
    }

    #[test]
    fn point2_index() {
        let p1 = Point2i::new(0, 1);

        assert_eq!(p1[0], 0);
        assert_eq!(p1[1], 1);
    }

    #[test]
    fn point2_index_mut() {
        let mut p1 = Point2i::new(0, 1);
        p1[0] = 3;
        p1[1] = 4;

        assert_eq!(p1[0], 3);
        assert_eq!(p1[1], 4);
    }

    #[test]
    #[should_panic]
    fn point2_index_out_of_bounds() {
        let _ = Point2i::new(0, 1)[2];
    }

    #[test]
    #[should_panic]
    fn point2_index_mut_out_of_bounds() {
        let mut p1 = Point2i::new(0, 1);
        p1[2] = 2;
    }

    #[test]
    fn point2_permute() {
        let p1 = Point2i::new(1, 2);

        assert_eq!(p1.permute(1, 0), Point2i::new(2, 1));
    }

    #[test]
    fn point2_add() {
        let p1 = Point2i::new(2, 8);
        let p2 = Point2i::new(3, 5);
        let p3 = Point2f::new(r64(2.0), r64(8.0));
        let p4 = Point2f::new(r64(3.0), r64(5.0));

        assert_eq!(p1 + p2, Point2i::new(5, 13));
        assert_eq!(p3 + p4, Point2f::new(r64(5.0), r64(13.0)));
    }

    #[test]
    fn point2_add_with_vector3() {
        let p1 = Point2i::new(2, 8);
        let v1 = Vector2i::new(3, 5);
        let p2 = Point2f::new(r64(2.0), r64(8.0));
        let v2 = Vector2f::new(r64(3.0), r64(5.0));

        assert_eq!(p1 + v1, Point2i::new(5, 13));
        assert_eq!(p2 + v2, Point2f::new(r64(5.0), r64(13.0)));
    }

    #[test]
    fn point2_sub() {
        let p1 = Point2i::new(2, 8);
        let p2 = Point2i::new(3, 5);
        let p3 = Point2f::new(r64(2.0), r64(8.0));
        let p4 = Point2f::new(r64(3.0), r64(5.0));

        assert_eq!(p1 - p2, Vector2i::new(-1, 3));
        assert_eq!(p3 - p4, Vector2f::new(r64(-1.0), r64(3.0)));
    }

    #[test]
    fn point2_sub_with_vector3() {
        let p1 = Point2i::new(2, 8);
        let v1 = Vector2i::new(3, 5);
        let p2 = Point2f::new(r64(2.0), r64(8.0));
        let v2 = Vector2f::new(r64(3.0), r64(5.0));

        assert_eq!(p1 - v1, Point2i::new(-1, 3));
        assert_eq!(p2 - v2, Point2f::new(r64(-1.0), r64(3.0)));
    }

    #[test]
    fn point2_mul() {
        let p1 = Point2i::new(2, 8);
        let p2 = Point2f::new(r64(2.0), r64(8.0));

        assert_eq!(p1 * 2, Point2i::new(4, 16));
        assert_eq!(p2 * r64(2.0), Point2f::new(r64(4.0), r64(16.0)));
    }

    #[test]
    fn point2_mul_assign() {
        let mut p1 = Point2i::new(2, 8);
        p1 *= 2;

        let mut p2 = Point2f::new(r64(2.0), r64(8.0));
        p2 *= r64(2.0);

        assert_eq!(p1, Point2i::new(4, 16));
        assert_eq!(p2, Point2f::new(r64(4.0), r64(16.0)));
    }

    #[test]
    fn point2_div() {
        let p1 = Point2i::new(2, 8);
        let p2 = Point2f::new(r64(2.0), r64(8.0));

        assert_eq!(p1 / 2, Point2i::new(1, 4));
        assert_eq!(p2 / r64(2.0), Point2f::new(r64(1.0), r64(4.0)));
    }

    #[test]
    fn point2_div_assign() {
        let mut p1 = Point2i::new(2, 8);
        p1 /= 2;

        let mut p2 = Point2f::new(r64(2.0), r64(8.0));
        p2 /= r64(2.0);

        assert_eq!(p1, Point2i::new(1, 4));
        assert_eq!(p2, Point2f::new(r64(1.0), r64(4.0)));
    }

    #[test]
    fn point2_neg() {
        let p1 = Point2i::new(-1, 2);
        let p2 = Point2f::new(r64(-1.0), r64(2.0));

        assert_eq!(-p1, Point2i::new(1, -2));
        assert_eq!(-p2, Point2f::new(r64(1.0), r64(-2.0)));
    }

    #[test]
    fn point2_distance_squared() {
        let p1 = Point2f::new(r64(1.0), r64(2.0));
        let p2 = Point2f::new(r64(0.0), r64(0.0));

        assert_eq!(p1.distance_squared(p2), r64(5.0));
    }

    #[test]
    fn point2_distance() {
        let p1 = Point2f::new(r64(1.0), r64(2.0));
        let p2 = Point2f::new(r64(0.0), r64(0.0));

        assert_eq!(p1.distance(p2), r64(5.0).sqrt());
    }

    #[test]
    fn point2_lerp3() {
        let p1 = Point2f::new(r64(0.0), r64(0.0));
        let p2 = Point2f::new(r64(1.0), r64(1.0));

        assert_eq!(lerp2(r64(0.5), p1, p2), Point2f::new(r64(0.5), r64(0.5)));
    }

    // Point3 tests

    #[test]
    fn point3_to_point2() {
        let p1 = Point3i::new(1, 2, 3);

        assert_eq!(p1.to_point2(), Point2i::new(1, 2));
    }

    #[test]
    fn point3_min_component() {
        let p1 = Point3i::new(1, 2, 3);
        let p2 = Point3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(p1.min_component(), 1);
        assert_eq!(p2.min_component(), r64(1.0));
    }

    #[test]
    fn point3_max_component() {
        let p1 = Point3i::new(1, 2, 3);
        let p2 = Point3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(p1.max_component(), 3);
        assert_eq!(p2.max_component(), r64(3.0));
    }

    #[test]
    fn point3_min_dimension() {
        let p1 = Point3i::new(1, 2, 3);
        let p2 = Point3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(p1.min_dimension(), 0);
        assert_eq!(p2.min_dimension(), 0);
    }

    #[test]
    fn point3_max_dimension() {
        let p1 = Point3i::new(1, 2, 3);
        let p2 = Point3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(p1.max_dimension(), 2);
        assert_eq!(p2.max_dimension(), 2);
    }

    #[test]
    fn point3_min() {
        let p1 = Point3i::new(2, 8, 4);
        let p2 = Point3i::new(3, 5, 6);
        let p3 = Point3f::new(r64(2.0), r64(8.0), r64(4.0));
        let p4 = Point3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(p1.min(p2), Point3i::new(2, 5, 4));
        assert_eq!(p3.min(p4), Point3f::new(r64(2.0), r64(5.0), r64(4.0)));
    }

    #[test]
    fn point3_max() {
        let p1 = Point3i::new(2, 8, 4);
        let p2 = Point3i::new(3, 5, 6);
        let p3 = Point3f::new(r64(2.0), r64(8.0), r64(4.0));
        let p4 = Point3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(p1.max(p2), Point3i::new(3, 8, 6));
        assert_eq!(p3.max(p4), Point3f::new(r64(3.0), r64(8.0), r64(6.0)));
    }

    #[test]
    fn point3_abs() {
        let p1 = Point3i::new(-1, 2, -4);
        let p2 = Point3f::new(r64(-1.0), r64(2.0), r64(-4.0));

        assert_eq!(p1.abs(), Point3i::new(1, 2, 4));
        assert_eq!(p2.abs(), Point3f::new(r64(1.0), r64(2.0), r64(4.0)));
    }

    #[test]
    fn point3_floor() {
        let p1 = Point3f::new(r64(1.1), r64(4.2), r64(-2.1));

        assert_eq!(p1.floor(), Point3f::new(r64(1.0), r64(4.0), r64(-3.0)));
    }

    #[test]
    fn point3_ceil() {
        let p1 = Point3f::new(r64(1.1), r64(4.2), r64(-2.1));

        assert_eq!(p1.ceil(), Point3f::new(r64(2.0), r64(5.0), r64(-2.0)));
    }

    #[test]
    fn point3_index() {
        let p1 = Point3i::new(0, 1, 2);

        assert_eq!(p1[0], 0);
        assert_eq!(p1[1], 1);
        assert_eq!(p1[2], 2);
    }

    #[test]
    fn point3_index_mut() {
        let mut p1 = Point3i::new(0, 1, 2);
        p1[0] = 3;
        p1[1] = 4;
        p1[2] = 5;

        assert_eq!(p1[0], 3);
        assert_eq!(p1[1], 4);
        assert_eq!(p1[2], 5);
    }

    #[test]
    #[should_panic]
    fn point3_index_out_of_bounds() {
        let _ = Point3i::new(0, 1, 2)[3];
    }

    #[test]
    #[should_panic]
    fn point3_index_mut_out_of_bounds() {
        let mut p1 = Point3i::new(0, 1, 2);
        p1[3] = 3;
    }

    #[test]
    fn point3_permute() {
        let p1 = Point3i::new(1, 2, 3);

        assert_eq!(p1.permute(2, 1, 0), Point3i::new(3, 2, 1));
    }

    #[test]
    fn point3_add() {
        let p1 = Point3i::new(2, 8, 4);
        let p2 = Point3i::new(3, 5, 6);
        let p3 = Point3f::new(r64(2.0), r64(8.0), r64(4.0));
        let p4 = Point3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(p1 + p2, Point3i::new(5, 13, 10));
        assert_eq!(p3 + p4, Point3f::new(r64(5.0), r64(13.0), r64(10.0)));
    }

    #[test]
    fn point3_add_with_vector3() {
        let p1 = Point3i::new(2, 8, 4);
        let v1 = Vector3i::new(3, 5, 6);
        let p2 = Point3f::new(r64(2.0), r64(8.0), r64(4.0));
        let v2 = Vector3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(p1 + v1, Point3i::new(5, 13, 10));
        assert_eq!(p2 + v2, Point3f::new(r64(5.0), r64(13.0), r64(10.0)));
    }

    #[test]
    fn point3_sub() {
        let p1 = Point3i::new(2, 8, 4);
        let p2 = Point3i::new(3, 5, 6);
        let p3 = Point3f::new(r64(2.0), r64(8.0), r64(4.0));
        let p4 = Point3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(p1 - p2, Vector3i::new(-1, 3, -2));
        assert_eq!(p3 - p4, Vector3f::new(r64(-1.0), r64(3.0), r64(-2.0)));
    }

    #[test]
    fn point3_sub_with_vector3() {
        let p1 = Point3i::new(2, 8, 4);
        let v1 = Vector3i::new(3, 5, 6);
        let p2 = Point3f::new(r64(2.0), r64(8.0), r64(4.0));
        let v2 = Vector3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(p1 - v1, Point3i::new(-1, 3, -2));
        assert_eq!(p2 - v2, Point3f::new(r64(-1.0), r64(3.0), r64(-2.0)));
    }

    #[test]
    fn point3_mul() {
        let p1 = Point3i::new(2, 8, 4);
        let p2 = Point3f::new(r64(2.0), r64(8.0), r64(4.0));

        assert_eq!(p1 * 2, Point3i::new(4, 16, 8));
        assert_eq!(p2 * r64(2.0), Point3f::new(r64(4.0), r64(16.0), r64(8.0)));
    }

    #[test]
    fn point3_mul_assign() {
        let mut p1 = Point3i::new(2, 8, 4);
        p1 *= 2;

        let mut p2 = Point3f::new(r64(2.0), r64(8.0), r64(4.0));
        p2 *= r64(2.0);

        assert_eq!(p1, Point3i::new(4, 16, 8));
        assert_eq!(p2, Point3f::new(r64(4.0), r64(16.0), r64(8.0)));
    }

    #[test]
    fn point3_div() {
        let p1 = Point3i::new(2, 8, 4);
        let p2 = Point3f::new(r64(2.0), r64(8.0), r64(4.0));

        assert_eq!(p1 / 2, Point3i::new(1, 4, 2));
        assert_eq!(p2 / r64(2.0), Point3f::new(r64(1.0), r64(4.0), r64(2.0)));
    }

    #[test]
    fn point3_div_assign() {
        let mut p1 = Point3i::new(2, 8, 4);
        p1 /= 2;

        let mut p2 = Point3f::new(r64(2.0), r64(8.0), r64(4.0));
        p2 /= r64(2.0);

        assert_eq!(p1, Point3i::new(1, 4, 2));
        assert_eq!(p2, Point3f::new(r64(1.0), r64(4.0), r64(2.0)));
    }

    #[test]
    fn point3_neg() {
        let p1 = Point3i::new(-1, 2, -3);
        let p2 = Point3f::new(r64(-1.0), r64(2.0), r64(-3.0));

        assert_eq!(-p1, Point3i::new(1, -2, 3));
        assert_eq!(-p2, Point3f::new(r64(1.0), r64(-2.0), r64(3.0)));
    }

    #[test]
    fn point3_distance_squared() {
        let p1 = Point3f::new(r64(1.0), r64(2.0), r64(3.0));
        let p2 = Point3f::new(r64(0.0), r64(0.0), r64(0.0));

        assert_eq!(p1.distance_squared(p2), r64(14.0));
    }

    #[test]
    fn point3_distance() {
        let p1 = Point3f::new(r64(1.0), r64(2.0), r64(3.0));
        let p2 = Point3f::new(r64(0.0), r64(0.0), r64(0.0));

        assert_eq!(p1.distance(p2), r64(14.0).sqrt());
    }

    #[test]
    fn point3_lerp3() {
        let p1 = Point3f::new(r64(0.0), r64(0.0), r64(0.0));
        let p2 = Point3f::new(r64(1.0), r64(1.0), r64(1.0));

        assert_eq!(
            lerp3(r64(0.5), p1, p2),
            Point3f::new(r64(0.5), r64(0.5), r64(0.5))
        );
    }
}
