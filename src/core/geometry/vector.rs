//! Two and three-dimensional vector representations.
//!
//! Generic implementations of two and three-dimensional vectors.
//! Convenience type aliases are provided for the isize and f64 item
//! types. A number of overloaded operators have been provided,
//! including:
//! * Element-wise addition
//! * Element-wise subtraction
//! * Multiplication by scalar (broadcasting, note that this is not
//! commutative - only vec * s is valid)
//! * Division by scalar (broadcasting, see above)
//!
//! Assignment equivalents of the above operations are also available,
//! as well as indexing operations.
//!
//! ```
//! # use ganymede::core::geometry::vector::*;
//! let mut a = Vec2i::new(1, 2);
//! let b = Vec2i::new(3, 4);
//!
//! let _ = a + b;
//! let _ = a - b;
//! let _ = a * 2;
//! let _ = a / 2;
//!
//! a += b;
//! a -= b;
//! a *= 2;
//! a /= 2;
//!
//! assert_eq!(a[0], 1); // a[0] == 1
//! ```

use num::{abs, Float, Integer, Num, Signed, Zero};
use std::cmp::{max, min};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

/// A trait implemented by all vectors.
pub trait Vec {
    type Item: VecItem;
}

/// A trait implemented by all vector items.
pub trait VecItem: Copy + Clone + Default + PartialEq {}

/// A trait for vectors that contain numbers.
pub trait VecNum: Vec
where
    Self::Item: VecItem + Num,
{
    fn dot(self, other: Self) -> Self::Item;
}

/// A trait for vectors that contain signed numbers.
pub trait VecSigned: Vec
where
    Self::Item: VecItem + Signed,
{
    fn dot_abs(self, other: Self) -> Self::Item;
}

/// A trait for vectors that contain integers.
pub trait VecInt: Vec
where
    Self::Item: VecItem + Integer,
{
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min_component(&self) -> Self::Item;
    fn max_component(&self) -> Self::Item;
}

/// A trait for vectors that contain floats.
pub trait VecFloat: Vec
where
    Self::Item: VecItem + Float,
{
    fn has_nans(&self) -> bool;
    fn length_squared(&self) -> Self::Item;
    fn length(&self) -> Self::Item;
    fn normalise(&self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min_component(&self) -> Self::Item;
    fn max_component(&self) -> Self::Item;
}

impl VecItem for isize {}
impl VecItem for f64 {}

/// A 2D vector of floats.
pub type Vec2f = Vec2<f64>;
/// A 2D vector of integers.
pub type Vec2i = Vec2<isize>;
/// A 3D vector of floats.
pub type Vec3f = Vec3<f64>;
/// A 3D vector of integers.
pub type Vec3i = Vec3<isize>;

/// A generic 2D vector.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct Vec2<T: VecItem> {
    pub x: T,
    pub y: T,
}

impl<T: VecItem> Vec2<T> {
    /// Instantiate a new two-dimensional vector.
    pub fn new(x: T, y: T) -> Vec2<T> {
        Vec2 { x, y }
    }
}

// Vec trait

impl<T: VecItem> Vec for Vec2<T> {
    type Item = T;
}

// VecNum trait

impl<T: VecItem + Num> VecNum for Vec2<T> {
    fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y
    }
}

// VecSigned trait

impl<T: VecItem + Signed> VecSigned for Vec2<T> {
    fn dot_abs(self, other: Self) -> T {
        abs(self.dot(other))
    }
}

// VecInt trait

impl<T: VecItem + Integer> VecInt for Vec2<T> {
    fn min(self, other: Self) -> Self {
        Vec2::new(min(self.x, other.x), min(self.y, other.y))
    }

    fn max(self, other: Self) -> Self {
        Vec2::new(max(self.x, other.x), max(self.y, other.y))
    }

    fn min_component(&self) -> T {
        min(self.x, self.y)
    }

    fn max_component(&self) -> T {
        max(self.x, self.y)
    }
}

// VecFloat trait

impl<T: VecItem + Float> VecFloat for Vec2<T> {
    /// Tests whether any elements of the Vector2 are NaNs or
    /// infinity.
    fn has_nans(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.x.is_infinite() || self.y.is_infinite()
    }

    fn length_squared(&self) -> Self::Item {
        self.x * self.x + self.y * self.y
    }

    fn length(&self) -> Self::Item {
        self.length_squared().sqrt()
    }

    fn normalise(&self) -> Self {
        *self / self.length()
    }

    fn min(self, other: Self) -> Self {
        Vec2::new(self.x.min(other.x), self.y.min(other.y))
    }

    fn max(self, other: Self) -> Self {
        Vec2::new(self.x.max(other.x), self.y.max(other.y))
    }

    fn min_component(&self) -> Self::Item {
        self.x.min(self.y)
    }

    fn max_component(&self) -> Self::Item {
        self.x.max(self.y)
    }
}

// Indexing traits

impl<T: VecItem> Index<u8> for Vec2<T> {
    type Output = T;

    fn index(&self, index: u8) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index out of bounds (Vec2)"),
        }
    }
}

impl<T: VecItem> IndexMut<u8> for Vec2<T> {
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index out of bounds (Vec2)"),
        }
    }
}

impl<T: VecItem> Vec2<T> {
    pub fn permute(&self, x: u8, y: u8) -> Vec2<T> {
        Vec2::new(self[x], self[y])
    }
}

// Operator traits

impl<T: VecItem + Add<Output = T>> Add for Vec2<T> {
    type Output = Vec2<T>;

    fn add(self, other: Vec2<T>) -> Self::Output {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<T: VecItem + AddAssign> AddAssign for Vec2<T> {
    fn add_assign(&mut self, other: Vec2<T>) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl<T: VecItem + Sub<Output = T>> Sub for Vec2<T> {
    type Output = Vec2<T>;

    fn sub(self, other: Vec2<T>) -> Self::Output {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<T: VecItem + SubAssign> SubAssign for Vec2<T> {
    fn sub_assign(&mut self, other: Vec2<T>) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl<T: VecItem + Mul<T, Output = T>> Mul<T> for Vec2<T> {
    type Output = Vec2<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Vec2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T: VecItem + MulAssign> MulAssign<T> for Vec2<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl<T: VecItem + Div<T, Output = T>> Div<T> for Vec2<T> {
    type Output = Vec2<T>;

    fn div(self, rhs: T) -> Self::Output {
        Vec2 {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl<T: VecItem + DivAssign> DivAssign<T> for Vec2<T> {
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

/// A generic 3D vector.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct Vec3<T: VecItem> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: VecItem> Vec3<T> {
    /// Instantiate a new three-dimensional vector.
    pub fn new(x: T, y: T, z: T) -> Vec3<T> {
        Vec3 { x, y, z }
    }
}

// Vec trait

impl<T: VecItem> Vec for Vec3<T> {
    type Item = T;
}

// VecNum trait

impl<T: VecItem + Num> VecNum for Vec3<T> {
    fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

// VecSigned trait

impl<T: VecItem + Signed> VecSigned for Vec3<T> {
    fn dot_abs(self, other: Self) -> T {
        abs(self.dot(other))
    }
}

// VecInt trait

impl<T: VecItem + Integer> VecInt for Vec3<T> {
    fn min(self, other: Self) -> Self {
        Vec3::new(
            min(self.x, other.x),
            min(self.y, other.y),
            min(self.z, other.z),
        )
    }

    fn max(self, other: Self) -> Self {
        Vec3::new(
            max(self.x, other.x),
            max(self.y, other.y),
            max(self.z, other.z),
        )
    }

    fn min_component(&self) -> Self::Item {
        min(self.x, min(self.y, self.z))
    }

    fn max_component(&self) -> Self::Item {
        max(self.x, max(self.y, self.z))
    }
}

// VecFloat trait

impl<T: VecItem + Float> VecFloat for Vec3<T> {
    /// Tests whether any elements of the Vector2 are NaNs or
    /// infinity.
    fn has_nans(&self) -> bool {
        self.x.is_nan()
            || self.y.is_nan()
            || self.z.is_nan()
            || self.x.is_infinite()
            || self.y.is_infinite()
            || self.z.is_infinite()
    }

    fn length_squared(&self) -> Self::Item {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn length(&self) -> Self::Item {
        self.length_squared().sqrt()
    }

    fn normalise(&self) -> Self {
        *self / self.length()
    }

    fn min(self, other: Self) -> Self {
        Vec3::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    fn max(self, other: Self) -> Self {
        Vec3::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    fn min_component(&self) -> Self::Item {
        self.x.min(self.y.min(self.z))
    }

    fn max_component(&self) -> Self::Item {
        self.x.max(self.y.max(self.z))
    }
}

// Cross product implementation

impl<T: VecItem + Num> Vec3<T> {
    pub fn cross(self, rhs: Self) -> Self {
        Vec3::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
}

// Coordinate system implementation

impl<T: VecItem + Float> Vec3<T> {
    pub fn coordinate_system(self) -> (Self, Self, Self) {
        let v1 = self.normalise();
        let v2 = match v1.x.abs() > v1.y.abs() {
            true => Vec3::new(-v1.z, Zero::zero(), v1.x).normalise(),
            false => Vec3::new(Zero::zero(), v1.z, -v1.y).normalise(),
        };
        let v3 = v1.cross(v2);
        (v1, v2, v3)
    }
}

// Indexing traits

impl<T: VecItem> Index<u8> for Vec3<T> {
    type Output = T;

    fn index(&self, index: u8) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index out of bounds (Vec2)"),
        }
    }
}

impl<T: VecItem> IndexMut<u8> for Vec3<T> {
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index out of bounds (Vec2)"),
        }
    }
}

impl<T: VecItem> Vec3<T> {
    pub fn permute(&self, x: u8, y: u8, z: u8) -> Vec3<T> {
        Vec3::new(self[x], self[y], self[z])
    }
}

// Operator traits

impl<T: VecItem + Add<Output = T>> Add for Vec3<T> {
    type Output = Vec3<T>;

    fn add(self, other: Vec3<T>) -> Self::Output {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: VecItem + AddAssign> AddAssign for Vec3<T> {
    fn add_assign(&mut self, other: Vec3<T>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<T: VecItem + Sub<Output = T>> Sub for Vec3<T> {
    type Output = Vec3<T>;

    fn sub(self, other: Vec3<T>) -> Self::Output {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: VecItem + SubAssign> SubAssign for Vec3<T> {
    fn sub_assign(&mut self, other: Vec3<T>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<T: VecItem + Mul<T, Output = T>> Mul<T> for Vec3<T> {
    type Output = Vec3<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T: VecItem + MulAssign> MulAssign<T> for Vec3<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl<T: VecItem + Div<T, Output = T>> Div<T> for Vec3<T> {
    type Output = Vec3<T>;

    fn div(self, rhs: T) -> Self::Output {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T: VecItem + DivAssign> DivAssign<T> for Vec3<T> {
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

#[cfg(test)]
#[cfg_attr(tarpaulin, skip)]
mod tests {
    use super::*;

    use std;

    // Vec2 tests

    #[test]
    fn vec2_indexing() {
        let a = Vec2i::new(0, 1);
        assert_eq!(a[1], 1);

        let mut a_mut = Vec2i::new(0, 1);
        a_mut[1] = 2;
        assert_eq!(a_mut[1], 2);
    }

    #[test]
    #[should_panic]
    fn vec2_index_out_of_bounds() {
        let _ = Vec2i::new(0, 1)[2];
    }

    #[test]
    #[should_panic]
    fn vec2_index_mut_out_of_bounds() {
        let mut a_mut = Vec2i::new(0, 1);
        a_mut[2] = 2;
    }

    #[test]
    fn vec2_permute() {
        let a = Vec2i::new(1, 2);
        assert_eq!(a.permute(1, 0), Vec2i::new(2, 1));
    }

    #[test]
    fn vec2_basic_operators() {
        let a = Vec2i::new(0, 1);
        let b = Vec2i::new(2, 3);
        let f = Vec2f::new(5.0, 10.0);

        assert_eq!(a + b, Vec2i::new(2, 4));
        assert_eq!(a - b, Vec2i::new(-2, -2));
        assert_eq!(a * 2, Vec2i::new(0, 2));
        assert_eq!(a / 2, Vec2i::new(0, 0));
        assert_eq!(f / 2.0, Vec2f::new(2.5, 5.0));
    }

    #[test]
    fn vec2_assignment_operators() {
        let mut a = Vec2i::new(0, 1);
        let b = Vec2i::new(2, 3);

        a += b;
        assert_eq!(a, Vec2i::new(2, 4));
        a -= b;
        assert_eq!(a, Vec2i::new(0, 1));
        a *= 2;
        assert_eq!(a, Vec2i::new(0, 2));
        a /= 2;
        assert_eq!(a, Vec2i::new(0, 1));
    }

    #[test]
    fn vec2_has_nans() {
        let f1 = Vec2f::new(1.0, 2.0);
        let f2 = Vec2f::new(1.0, std::f64::NAN);
        let f3 = Vec2f::new(1.0, std::f64::INFINITY);

        assert!(!f1.has_nans());
        assert!(f2.has_nans());
        assert!(f3.has_nans());
    }

    #[test]
    fn vec2_dot() {
        let a = Vec2i::new(-1, 2);
        let b = Vec2i::new(3, -4);
        assert_eq!(a.dot(b), -11);
        assert_eq!(a.dot_abs(b), 11)
    }

    #[test]
    fn vec2_length() {
        let a = Vec2f::new(1.0, 2.0);
        assert_eq!(a.length(), 5.0.sqrt());
    }

    #[test]
    fn vec2_normalise() {
        let a = Vec2f::new(2.0, 2.0);
        assert_eq!(
            a.normalise(),
            Vec2f::new(1.0 / 2.0.sqrt(), 1.0 / 2.0.sqrt())
        );
    }

    #[test]
    fn vec2_min() {
        let a = Vec2i::new(1, 10);
        let b = Vec2i::new(2, 4);
        assert_eq!(a.min(b), Vec2i::new(1, 4));

        let c = Vec2f::new(1.0, 10.0);
        let d = Vec2f::new(2.0, 4.0);
        assert_eq!(c.min(d), Vec2f::new(1.0, 4.0));
    }

    #[test]
    fn vec2_max() {
        let a = Vec2i::new(1, 10);
        let b = Vec2i::new(2, 4);
        assert_eq!(a.max(b), Vec2i::new(2, 10));

        let c = Vec2f::new(1.0, 10.0);
        let d = Vec2f::new(2.0, 4.0);
        assert_eq!(c.max(d), Vec2f::new(2.0, 10.0));
    }

    #[test]
    fn vec2_min_component() {
        let a = Vec2i::new(1, 2);
        let b = Vec2f::new(1.0, 2.0);
        assert_eq!(a.min_component(), 1);
        assert_eq!(b.min_component(), 1.0);
    }

    #[test]
    fn vec2_max_component() {
        let a = Vec2i::new(1, 2);
        let b = Vec2f::new(1.0, 2.0);
        assert_eq!(a.max_component(), 2);
        assert_eq!(b.max_component(), 2.0);
    }

    // Vec3 tests

    #[test]
    fn vec3_indexing() {
        let a = Vec3i::new(0, 1, 2);
        assert_eq!(a[2], 2);

        let mut a_mut = Vec3i::new(0, 1, 2);
        a_mut[2] = 3;
        assert_eq!(a_mut[2], 3);
    }

    #[test]
    #[should_panic]
    fn vec3_index_out_of_bounds() {
        let _ = Vec3i::new(0, 1, 2)[3];
    }

    #[test]
    #[should_panic]
    fn vec3_index_mut_out_of_bounds() {
        let mut a_mut = Vec3i::new(0, 1, 2);
        a_mut[3] = 3;
    }

    #[test]
    fn vec3_permute() {
        let a = Vec3i::new(1, 2, 3);
        assert_eq!(a.permute(2, 1, 0), Vec3i::new(3, 2, 1));
    }

    #[test]
    fn vec3_basic_operators() {
        let a = Vec3i::new(0, 1, 2);
        let b = Vec3i::new(3, 4, 5);
        let c = Vec3f::new(5.0, 10.0, 20.0);

        assert_eq!(a + b, Vec3i::new(3, 5, 7));
        assert_eq!(a - b, Vec3i::new(-3, -3, -3));
        assert_eq!(a * 2, Vec3i::new(0, 2, 4));
        assert_eq!(a / 2, Vec3i::new(0, 0, 1));
        assert_eq!(c / 2.0, Vec3f::new(2.5, 5.0, 10.0));
    }

    #[test]
    fn vec3_assignment_operators() {
        let mut a = Vec3i::new(0, 1, 2);
        let b = Vec3i::new(3, 4, 5);

        a += b;
        assert_eq!(a, Vec3i::new(3, 5, 7));
        a -= b;
        assert_eq!(a, Vec3i::new(0, 1, 2));
        a *= 2;
        assert_eq!(a, Vec3i::new(0, 2, 4));
        a /= 2;
        assert_eq!(a, Vec3i::new(0, 1, 2));
    }

    #[test]
    fn vec3_has_nans() {
        let f1 = Vec3f::new(1.0, 2.0, 3.0);
        let f2 = Vec3f::new(1.0, std::f64::NAN, 3.0);
        let f3 = Vec3f::new(1.0, std::f64::INFINITY, 3.0);

        assert!(!f1.has_nans());
        assert!(f2.has_nans());
        assert!(f3.has_nans());
    }

    #[test]
    fn vec3_dot() {
        let a = Vec3i::new(-1, 2, -3);
        let b = Vec3i::new(4, -5, 6);
        assert_eq!(a.dot(b), -32);
        assert_eq!(a.dot_abs(b), 32)
    }

    #[test]
    fn vec3_cross() {
        let a = Vec3i::new(1, 2, 3);
        let b = Vec3i::new(4, 5, 6);
        assert_eq!(a.cross(b), Vec3::new(-3, 6, -3));
    }

    #[test]
    fn vec3_length() {
        let a = Vec3f::new(1.0, 2.0, 3.0);
        assert_eq!(a.length(), 14.0.sqrt())
    }

    #[test]
    fn vec3_normalise() {
        let a = Vec3f::new(2.0, 2.0, 2.0);
        assert_eq!(
            a.normalise(),
            Vec3f::new(1.0 / 3.0.sqrt(), 1.0 / 3.0.sqrt(), 1.0 / 3.0.sqrt())
        );
    }

    #[test]
    fn vec3_min() {
        let a = Vec3i::new(1, 10, 3);
        let b = Vec3i::new(2, 4, 7);
        assert_eq!(a.min(b), Vec3i::new(1, 4, 3));

        let c = Vec3f::new(1.0, 10.0, 3.0);
        let d = Vec3f::new(2.0, 4.0, 7.0);
        assert_eq!(c.min(d), Vec3f::new(1.0, 4.0, 3.0));
    }

    #[test]
    fn vec3_max() {
        let a = Vec3i::new(1, 10, 3);
        let b = Vec3i::new(2, 4, 7);
        assert_eq!(a.max(b), Vec3i::new(2, 10, 7));

        let c = Vec3f::new(1.0, 10.0, 3.0);
        let d = Vec3f::new(2.0, 4.0, 7.0);
        assert_eq!(c.max(d), Vec3f::new(2.0, 10.0, 7.0));
    }

    #[test]
    fn vec3_min_component() {
        let a = Vec3i::new(1, 2, 3);
        let b = Vec3f::new(1.0, 2.0, 3.0);
        assert_eq!(a.min_component(), 1);
        assert_eq!(b.min_component(), 1.0);
    }

    #[test]
    fn vec3_max_component() {
        let a = Vec3i::new(1, 2, 3);
        let b = Vec3f::new(1.0, 2.0, 3.0);
        assert_eq!(a.max_component(), 3);
        assert_eq!(b.max_component(), 3.0);
    }

    #[test]
    fn coordinate_system() {
        let a = Vec3f::new(1.0, 0.0, 0.0);
        assert_eq!(
            a.coordinate_system(), 
            (
                Vec3f::new(1.0, 0.0, 0.0),
                Vec3f::new(0.0, 0.0, 1.0),
                Vec3f::new(0.0, -1.0, 0.0),
            )
        );

        let b = Vec3f::new(0.0, 1.0, 0.0);
        assert_eq!(
            b.coordinate_system(), 
            (
                Vec3f::new(0.0, 1.0, 0.0),
                Vec3f::new(0.0, 0.0, -1.0),
                Vec3f::new(-1.0, 0.0, 0.0),
            )
        );
    }
}
