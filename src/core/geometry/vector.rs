//! Vector representations for two and three dimensions.

use std::cmp::{max, min};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use noisy_float::prelude::*;
use num::{abs, Float, Signed, Zero};

use super::normal::{Normal2, Normal3};
use super::ops::{Cross, Dot, FaceForward, Length, Normalise};
use super::{Prim, PrimFloat, PrimItem, PrimSigned};

/// Three-dimensional vector of integers.
pub type Vector2i = Vector2<isize>;
/// Three-dimensional vector of floats.
pub type Vector2f = Vector2<R64>;
/// Three-dimensional vector of integers.
pub type Vector3i = Vector3<isize>;
/// Three-dimensional vector of floats.
pub type Vector3f = Vector3<R64>;

/// A generic, two-dimensional vector.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct Vector2<T: PrimItem> {
    pub x: T,
    pub y: T,
}

// Prim trait implementation

impl<T: PrimItem> Prim for Vector2<T> {
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
        Vector2 {
            x: min(self.x, other.x),
            y: min(self.y, other.y),
        }
    }

    fn max(self, other: Self) -> Self {
        Vector2 {
            x: max(self.x, other.x),
            y: max(self.y, other.y),
        }
    }
}

// PrimSigned trait implementation

impl<T: PrimItem + Signed> PrimSigned for Vector2<T> {
    fn abs(self) -> Self {
        Vector2 {
            x: abs(self.x),
            y: abs(self.y),
        }
    }
}

// PrimFloat trait implementation

impl<T: PrimItem + Float> PrimFloat for Vector2<T> {
    fn floor(self) -> Self {
        Vector2 {
            x: self.x.floor(),
            y: self.y.floor(),
        }
    }

    fn ceil(self) -> Self {
        Vector2 {
            x: self.x.ceil(),
            y: self.y.ceil(),
        }
    }
}

// Constructors

impl<T: PrimItem> Vector2<T> {
    /// Construct a new Vector2 from individual component values.
    pub fn new(x: T, y: T) -> Self {
        Vector2 { x, y }
    }

    /// Construct a new Normal2 from a Vector2.
    pub fn to_normal(self) -> Normal2<T> {
        Normal2 {
            x: self.x,
            y: self.y,
        }
    }
}

// Indexing traits

impl<T: PrimItem> Index<usize> for Vector2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index out of bounds (Vector2)"),
        }
    }
}

impl<T: PrimItem> IndexMut<usize> for Vector2<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index out of bounds (Vector2)"),
        }
    }
}

impl<T: PrimItem> Vector2<T> {
    pub fn permute(&self, x: usize, y: usize) -> Vector2<T> {
        Vector2 {
            x: self[x],
            y: self[y],
        }
    }
}

// Operator traits

impl<T: PrimItem + Add<Output = T>> Add for Vector2<T> {
    type Output = Vector2<T>;

    fn add(self, other: Vector2<T>) -> Self::Output {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<T: PrimItem + AddAssign> AddAssign for Vector2<T> {
    fn add_assign(&mut self, other: Vector2<T>) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl<T: PrimItem + Sub<Output = T>> Sub for Vector2<T> {
    type Output = Vector2<T>;

    fn sub(self, other: Vector2<T>) -> Self::Output {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<T: PrimItem + SubAssign> SubAssign for Vector2<T> {
    fn sub_assign(&mut self, other: Vector2<T>) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl<T: PrimItem + Mul<T, Output = T>> Mul<T> for Vector2<T> {
    type Output = Vector2<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Vector2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T: PrimItem + MulAssign> MulAssign<T> for Vector2<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl<T: PrimItem + Div<T, Output = T>> Div<T> for Vector2<T> {
    type Output = Vector2<T>;

    fn div(self, rhs: T) -> Self::Output {
        Vector2 {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl<T: PrimItem + DivAssign> DivAssign<T> for Vector2<T> {
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl<T: PrimItem + Signed> Neg for Vector2<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Vector2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

// Length operation

impl<T: PrimItem + Float> Length for Vector2<T> {
    fn length_squared(self) -> Self::Item {
        self.x * self.x + self.y * self.y
    }
}

// Normalise operation

impl<T: PrimItem + Float> Normalise for Vector2<T> {
    fn normalise(self) -> Self {
        self / self.length()
    }
}

// Dot operation

impl<T: PrimItem + Signed> Dot for Vector2<T> {
    fn dot(self, rhs: Self) -> Self::Item {
        self.x * rhs.x + self.y * rhs.y
    }
}

impl<T: PrimItem + Signed> Dot<Normal2<T>> for Vector2<T> {
    fn dot(self, rhs: Normal2<T>) -> Self::Item {
        self.x * rhs.x + self.y * rhs.y
    }
}

/// A generic, three-dimensional vector.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct Vector3<T: PrimItem> {
    pub x: T,
    pub y: T,
    pub z: T,
}

// Prim trait implementation

impl<T: PrimItem> Prim for Vector3<T> {
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
        Vector3 {
            x: min(self.x, other.x),
            y: min(self.y, other.y),
            z: min(self.z, other.z),
        }
    }

    fn max(self, other: Self) -> Self {
        Vector3 {
            x: max(self.x, other.x),
            y: max(self.y, other.y),
            z: max(self.z, other.z),
        }
    }
}

// PrimSigned trait implementation

impl<T: PrimItem + Signed> PrimSigned for Vector3<T> {
    fn abs(self) -> Self {
        Vector3 {
            x: abs(self.x),
            y: abs(self.y),
            z: abs(self.z),
        }
    }
}

// PrimFloat trait implementation

impl<T: PrimItem + Float> PrimFloat for Vector3<T> {
    fn floor(self) -> Self {
        Vector3 {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
        }
    }

    fn ceil(self) -> Self {
        Vector3 {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
        }
    }
}

// Constructors

impl<T: PrimItem> Vector3<T> {
    /// Construct a new Vector3 from individual component values.
    pub fn new(x: T, y: T, z: T) -> Self {
        Vector3 { x, y, z }
    }

    /// Construct a new Normal3 from a Vector3.
    pub fn to_normal(self) -> Normal3<T> {
        Normal3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

// Indexing traits

impl<T: PrimItem> Index<usize> for Vector3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index out of bounds (Vector3)"),
        }
    }
}

impl<T: PrimItem> IndexMut<usize> for Vector3<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index out of bounds (Vector3)"),
        }
    }
}

impl<T: PrimItem> Vector3<T> {
    pub fn permute(&self, x: usize, y: usize, z: usize) -> Vector3<T> {
        Vector3 {
            x: self[x],
            y: self[y],
            z: self[z],
        }
    }
}

// Operator traits

impl<T: PrimItem + Add<Output = T>> Add for Vector3<T> {
    type Output = Vector3<T>;

    fn add(self, other: Vector3<T>) -> Self::Output {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: PrimItem + AddAssign> AddAssign for Vector3<T> {
    fn add_assign(&mut self, other: Vector3<T>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<T: PrimItem + Sub<Output = T>> Sub for Vector3<T> {
    type Output = Vector3<T>;

    fn sub(self, other: Vector3<T>) -> Self::Output {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: PrimItem + SubAssign> SubAssign for Vector3<T> {
    fn sub_assign(&mut self, other: Vector3<T>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<T: PrimItem + Mul<T, Output = T>> Mul<T> for Vector3<T> {
    type Output = Vector3<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Vector3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T: PrimItem + MulAssign> MulAssign<T> for Vector3<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl<T: PrimItem + Div<T, Output = T>> Div<T> for Vector3<T> {
    type Output = Vector3<T>;

    fn div(self, rhs: T) -> Self::Output {
        Vector3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T: PrimItem + DivAssign> DivAssign<T> for Vector3<T> {
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl<T: PrimItem + Signed> Neg for Vector3<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Vector3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// Length operation

impl<T: PrimItem + Float> Length for Vector3<T> {
    fn length_squared(self) -> Self::Item {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}

// Normalise operation

impl<T: PrimItem + Float> Normalise for Vector3<T> {
    fn normalise(self) -> Self {
        self / self.length()
    }
}

// Dot operation

impl<T: PrimItem + Signed> Dot for Vector3<T> {
    fn dot(self, rhs: Self) -> Self::Item {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl<T: PrimItem + Signed> Dot<Normal3<T>> for Vector3<T> {
    fn dot(self, rhs: Normal3<T>) -> Self::Item {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

// Cross operation

impl<T: PrimItem + Signed> Cross for Vector3<T> {
    type Output = Self;

    fn cross(self, rhs: Self) -> Self::Output {
        Vector3::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
}

impl<T: PrimItem + Signed> Cross<Normal3<T>> for Vector3<T> {
    type Output = Self;

    fn cross(self, rhs: Normal3<T>) -> Self::Output {
        Vector3::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
}

// Face-forward operation

impl<T: PrimItem + Signed> FaceForward for Vector3<T> {
    type Output = Self;

    fn face_forward(self, rhs: Self) -> Self::Output {
        match self.dot(rhs) < Zero::zero() {
            true => -self,
            false => self,
        }
    }
}

impl<T: PrimItem + Signed> FaceForward<Normal3<T>> for Vector3<T> {
    type Output = Self;

    fn face_forward(self, rhs: Normal3<T>) -> Self::Output {
        match self.dot(rhs) < Zero::zero() {
            true => -self,
            false => self,
        }
    }
}

// Coordinate system generation

impl<T: PrimItem + Signed + Float> Vector3<T> {
    pub fn coordinate_system(self) -> (Self, Self, Self) {
        let v1 = self.normalise();
        let v2 = match v1.x.abs() > v1.y.abs() {
            true => Vector3::new(-v1.z, Zero::zero(), v1.x).normalise(),
            false => Vector3::new(Zero::zero(), v1.z, -v1.y).normalise(),
        };
        let v3 = v1.cross(v2);
        (v1, v2, v3)
    }
}

// Tests

#[cfg(test)]
#[cfg_attr(tarpaulin, skip)]
mod tests {
    use super::super::normal::{Normal2i, Normal3i};
    use super::*;

    // Vector2 tests

    #[test]
    fn vector2_to_normal2() {
        let v1 = Vector2i::new(1, 2);

        assert_eq!(v1.to_normal(), Normal2i::new(1, 2));
    }

    #[test]
    fn vector2_min_component() {
        let v1 = Vector2i::new(1, 2);
        let v2 = Vector2f::new(r64(1.0), r64(2.0));

        assert_eq!(v1.min_component(), 1);
        assert_eq!(v2.min_component(), r64(1.0));
    }

    #[test]
    fn vector2_max_component() {
        let v1 = Vector2i::new(1, 2);
        let v2 = Vector2f::new(r64(1.0), r64(2.0));

        assert_eq!(v1.max_component(), 2);
        assert_eq!(v2.max_component(), r64(2.0));
    }

    #[test]
    fn vector2_min_dimension() {
        let v1 = Vector2i::new(1, 2);
        let v2 = Vector2f::new(r64(1.0), r64(2.0));

        assert_eq!(v1.min_dimension(), 0);
        assert_eq!(v2.min_dimension(), 0);
    }

    #[test]
    fn vector2_max_dimension() {
        let v1 = Vector2i::new(1, 2);
        let v2 = Vector2f::new(r64(1.0), r64(2.0));

        assert_eq!(v1.max_dimension(), 1);
        assert_eq!(v2.max_dimension(), 1);
    }

    #[test]
    fn vector2_min() {
        let v1 = Vector2i::new(2, 8);
        let v2 = Vector2i::new(3, 5);
        let v3 = Vector2f::new(r64(2.0), r64(8.0));
        let v4 = Vector2f::new(r64(3.0), r64(5.0));

        assert_eq!(v1.min(v2), Vector2i::new(2, 5));
        assert_eq!(v3.min(v4), Vector2f::new(r64(2.0), r64(5.0)));
    }

    #[test]
    fn vector2_max() {
        let v1 = Vector2i::new(2, 8);
        let v2 = Vector2i::new(3, 5);
        let v3 = Vector2f::new(r64(2.0), r64(8.0));
        let v4 = Vector2f::new(r64(3.0), r64(5.0));

        assert_eq!(v1.max(v2), Vector2i::new(3, 8));
        assert_eq!(v3.max(v4), Vector2f::new(r64(3.0), r64(8.0)));
    }

    #[test]
    fn vector2_abs() {
        let v1 = Vector2i::new(-1, 2);
        let v2 = Vector2f::new(r64(-1.0), r64(2.0));

        assert_eq!(v1.abs(), Vector2i::new(1, 2));
        assert_eq!(v2.abs(), Vector2f::new(r64(1.0), r64(2.0)));
    }

    #[test]
    fn vector2_floor() {
        let v1 = Vector2f::new(r64(1.1), r64(4.2));

        assert_eq!(v1.floor(), Vector2f::new(r64(1.0), r64(4.0)));
    }

    #[test]
    fn vector2_ceil() {
        let v1 = Vector2f::new(r64(1.1), r64(4.2));

        assert_eq!(v1.ceil(), Vector2f::new(r64(2.0), r64(5.0)));
    }

    #[test]
    fn vector2_index() {
        let v1 = Vector2i::new(0, 1);

        assert_eq!(v1[0], 0);
        assert_eq!(v1[1], 1);
    }

    #[test]
    fn vector2_index_mut() {
        let mut v1 = Vector2i::new(0, 1);
        v1[0] = 3;
        v1[1] = 4;

        assert_eq!(v1[0], 3);
        assert_eq!(v1[1], 4);
    }

    #[test]
    #[should_panic]
    fn vector2_index_out_of_bounds() {
        let _ = Vector2i::new(0, 1)[2];
    }

    #[test]
    #[should_panic]
    fn vector2_index_mut_out_of_bounds() {
        let mut v1 = Vector2i::new(0, 1);
        v1[2] = 2;
    }

    #[test]
    fn vector2_permute() {
        let v1 = Vector2i::new(1, 2);

        assert_eq!(v1.permute(1, 0), Vector2i::new(2, 1));
    }

    #[test]
    fn vector2_add() {
        let v1 = Vector2i::new(2, 8);
        let v2 = Vector2i::new(3, 5);
        let v3 = Vector2f::new(r64(2.0), r64(8.0));
        let v4 = Vector2f::new(r64(3.0), r64(5.0));

        assert_eq!(v1 + v2, Vector2i::new(5, 13));
        assert_eq!(v3 + v4, Vector2f::new(r64(5.0), r64(13.0)));
    }

    #[test]
    fn vector2_add_assign() {
        let mut v1 = Vector2i::new(2, 8);
        let v2 = Vector2i::new(3, 5);
        v1 += v2;

        let mut v3 = Vector2f::new(r64(2.0), r64(8.0));
        let v4 = Vector2f::new(r64(3.0), r64(5.0));
        v3 += v4;

        assert_eq!(v1, Vector2i::new(5, 13));
        assert_eq!(v3, Vector2f::new(r64(5.0), r64(13.0)));
    }

    #[test]
    fn vector2_sub() {
        let v1 = Vector2i::new(2, 8);
        let v2 = Vector2i::new(3, 5);
        let v3 = Vector2f::new(r64(2.0), r64(8.0));
        let v4 = Vector2f::new(r64(3.0), r64(5.0));

        assert_eq!(v1 - v2, Vector2i::new(-1, 3));
        assert_eq!(v3 - v4, Vector2f::new(r64(-1.0), r64(3.0)));
    }

    #[test]
    fn vector2_sub_assign() {
        let mut v1 = Vector2i::new(2, 8);
        let v2 = Vector2i::new(3, 5);
        v1 -= v2;

        let mut v3 = Vector2f::new(r64(2.0), r64(8.0));
        let v4 = Vector2f::new(r64(3.0), r64(5.0));
        v3 -= v4;

        assert_eq!(v1, Vector2i::new(-1, 3));
        assert_eq!(v3, Vector2f::new(r64(-1.0), r64(3.0)));
    }

    #[test]
    fn vector2_mul() {
        let v1 = Vector2i::new(2, 8);
        let v2 = Vector2f::new(r64(2.0), r64(8.0));

        assert_eq!(v1 * 2, Vector2i::new(4, 16));
        assert_eq!(v2 * r64(2.0), Vector2f::new(r64(4.0), r64(16.0)));
    }

    #[test]
    fn vector2_mul_assign() {
        let mut v1 = Vector2i::new(2, 8);
        v1 *= 2;

        let mut v2 = Vector2f::new(r64(2.0), r64(8.0));
        v2 *= r64(2.0);

        assert_eq!(v1, Vector2i::new(4, 16));
        assert_eq!(v2, Vector2f::new(r64(4.0), r64(16.0)));
    }

    #[test]
    fn vector2_div() {
        let v1 = Vector2i::new(2, 8);
        let v2 = Vector2f::new(r64(2.0), r64(8.0));

        assert_eq!(v1 / 2, Vector2i::new(1, 4));
        assert_eq!(v2 / r64(2.0), Vector2f::new(r64(1.0), r64(4.0)));
    }

    #[test]
    fn vector2_div_assign() {
        let mut v1 = Vector2i::new(2, 8);
        v1 /= 2;

        let mut v2 = Vector2f::new(r64(2.0), r64(8.0));
        v2 /= r64(2.0);

        assert_eq!(v1, Vector2i::new(1, 4));
        assert_eq!(v2, Vector2f::new(r64(1.0), r64(4.0)));
    }

    #[test]
    fn vector2_neg() {
        let v1 = Vector2i::new(-1, 2);
        let v2 = Vector2f::new(r64(-1.0), r64(2.0));

        assert_eq!(-v1, Vector2i::new(1, -2));
        assert_eq!(-v2, Vector2f::new(r64(1.0), r64(-2.0)));
    }

    #[test]
    fn vector2_length_squared() {
        let v1 = Vector2f::new(r64(1.0), r64(2.0));

        assert_eq!(v1.length_squared(), r64(5.0));
    }

    #[test]
    fn vector2_length() {
        let v1 = Vector2f::new(r64(1.0), r64(2.0));

        assert_eq!(v1.length(), r64(5.0).sqrt());
    }

    #[test]
    fn vector2_normalise() {
        let v1 = Vector2f::new(r64(2.0), r64(2.0));
        assert_eq!(
            v1.normalise(),
            Vector2f::new(r64(1.0 / 2.0.sqrt()), r64(1.0 / 2.0.sqrt()))
        );
    }

    #[test]
    fn vector2_dot() {
        let v1 = Vector2i::new(-1, 2);
        let v2 = Vector2i::new(3, -4);

        assert_eq!(v1.dot(v2), -11);
        assert_eq!(v1.dot_abs(v2), 11);
    }

    #[test]
    fn vector2_dot_normal2() {
        let v1 = Vector2i::new(-1, 2);
        let n1 = Normal2i::new(3, -4);

        assert_eq!(v1.dot(n1), -11);
        assert_eq!(v1.dot_abs(n1), 11);
    }

    // Vector3 tests

    #[test]
    fn vector3_to_normal3() {
        let v1 = Vector3i::new(1, 2, 3);

        assert_eq!(v1.to_normal(), Normal3i::new(1, 2, 3));
    }

    #[test]
    fn vector3_min_component() {
        let v1 = Vector3i::new(1, 2, 3);
        let v2 = Vector3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(v1.min_component(), 1);
        assert_eq!(v2.min_component(), r64(1.0));
    }

    #[test]
    fn vector3_max_component() {
        let v1 = Vector3i::new(1, 2, 3);
        let v2 = Vector3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(v1.max_component(), 3);
        assert_eq!(v2.max_component(), r64(3.0));
    }

    #[test]
    fn vector3_min_dimension() {
        let v1 = Vector3i::new(1, 2, 3);
        let v2 = Vector3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(v1.min_dimension(), 0);
        assert_eq!(v2.min_dimension(), 0);
    }

    #[test]
    fn vector3_max_dimension() {
        let v1 = Vector3i::new(1, 2, 3);
        let v2 = Vector3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(v1.max_dimension(), 2);
        assert_eq!(v2.max_dimension(), 2);
    }

    #[test]
    fn vector3_min() {
        let v1 = Vector3i::new(2, 8, 4);
        let v2 = Vector3i::new(3, 5, 6);
        let v3 = Vector3f::new(r64(2.0), r64(8.0), r64(4.0));
        let v4 = Vector3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(v1.min(v2), Vector3i::new(2, 5, 4));
        assert_eq!(v3.min(v4), Vector3f::new(r64(2.0), r64(5.0), r64(4.0)));
    }

    #[test]
    fn vector3_max() {
        let v1 = Vector3i::new(2, 8, 4);
        let v2 = Vector3i::new(3, 5, 6);
        let v3 = Vector3f::new(r64(2.0), r64(8.0), r64(4.0));
        let v4 = Vector3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(v1.max(v2), Vector3i::new(3, 8, 6));
        assert_eq!(v3.max(v4), Vector3f::new(r64(3.0), r64(8.0), r64(6.0)));
    }

    #[test]
    fn vector3_abs() {
        let v1 = Vector3i::new(-1, 2, -4);
        let v2 = Vector3f::new(r64(-1.0), r64(2.0), r64(-4.0));

        assert_eq!(v1.abs(), Vector3i::new(1, 2, 4));
        assert_eq!(v2.abs(), Vector3f::new(r64(1.0), r64(2.0), r64(4.0)));
    }

    #[test]
    fn vector3_floor() {
        let v1 = Vector3f::new(r64(1.1), r64(4.2), r64(-2.1));

        assert_eq!(v1.floor(), Vector3f::new(r64(1.0), r64(4.0), r64(-3.0)));
    }

    #[test]
    fn vector3_ceil() {
        let v1 = Vector3f::new(r64(1.1), r64(4.2), r64(-2.1));

        assert_eq!(v1.ceil(), Vector3f::new(r64(2.0), r64(5.0), r64(-2.0)));
    }

    #[test]
    fn vector3_index() {
        let v1 = Vector3i::new(0, 1, 2);

        assert_eq!(v1[0], 0);
        assert_eq!(v1[1], 1);
        assert_eq!(v1[2], 2);
    }

    #[test]
    fn vector3_index_mut() {
        let mut v1 = Vector3i::new(0, 1, 2);
        v1[0] = 3;
        v1[1] = 4;
        v1[2] = 5;

        assert_eq!(v1[0], 3);
        assert_eq!(v1[1], 4);
        assert_eq!(v1[2], 5);
    }

    #[test]
    #[should_panic]
    fn vector3_index_out_of_bounds() {
        let _ = Vector3i::new(0, 1, 2)[3];
    }

    #[test]
    #[should_panic]
    fn vector3_index_mut_out_of_bounds() {
        let mut v1 = Vector3i::new(0, 1, 2);
        v1[3] = 3;
    }

    #[test]
    fn vector3_permute() {
        let v1 = Vector3i::new(1, 2, 3);

        assert_eq!(v1.permute(2, 1, 0), Vector3i::new(3, 2, 1));
    }

    #[test]
    fn vector3_add() {
        let v1 = Vector3i::new(2, 8, 4);
        let v2 = Vector3i::new(3, 5, 6);
        let v3 = Vector3f::new(r64(2.0), r64(8.0), r64(4.0));
        let v4 = Vector3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(v1 + v2, Vector3i::new(5, 13, 10));
        assert_eq!(v3 + v4, Vector3f::new(r64(5.0), r64(13.0), r64(10.0)));
    }

    #[test]
    fn vector3_add_assign() {
        let mut v1 = Vector3i::new(2, 8, 4);
        let v2 = Vector3i::new(3, 5, 6);
        v1 += v2;

        let mut v3 = Vector3f::new(r64(2.0), r64(8.0), r64(4.0));
        let v4 = Vector3f::new(r64(3.0), r64(5.0), r64(6.0));
        v3 += v4;

        assert_eq!(v1, Vector3i::new(5, 13, 10));
        assert_eq!(v3, Vector3f::new(r64(5.0), r64(13.0), r64(10.0)));
    }

    #[test]
    fn vector3_sub() {
        let v1 = Vector3i::new(2, 8, 4);
        let v2 = Vector3i::new(3, 5, 6);
        let v3 = Vector3f::new(r64(2.0), r64(8.0), r64(4.0));
        let v4 = Vector3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(v1 - v2, Vector3i::new(-1, 3, -2));
        assert_eq!(v3 - v4, Vector3f::new(r64(-1.0), r64(3.0), r64(-2.0)));
    }

    #[test]
    fn vector3_sub_assign() {
        let mut v1 = Vector3i::new(2, 8, 4);
        let v2 = Vector3i::new(3, 5, 6);
        v1 -= v2;

        let mut v3 = Vector3f::new(r64(2.0), r64(8.0), r64(4.0));
        let v4 = Vector3f::new(r64(3.0), r64(5.0), r64(6.0));
        v3 -= v4;

        assert_eq!(v1, Vector3i::new(-1, 3, -2));
        assert_eq!(v3, Vector3f::new(r64(-1.0), r64(3.0), r64(-2.0)));
    }

    #[test]
    fn vector3_mul() {
        let v1 = Vector3i::new(2, 8, 4);
        let v2 = Vector3f::new(r64(2.0), r64(8.0), r64(4.0));

        assert_eq!(v1 * 2, Vector3i::new(4, 16, 8));
        assert_eq!(v2 * r64(2.0), Vector3f::new(r64(4.0), r64(16.0), r64(8.0)));
    }

    #[test]
    fn vector3_mul_assign() {
        let mut v1 = Vector3i::new(2, 8, 4);
        v1 *= 2;

        let mut v2 = Vector3f::new(r64(2.0), r64(8.0), r64(4.0));
        v2 *= r64(2.0);

        assert_eq!(v1, Vector3i::new(4, 16, 8));
        assert_eq!(v2, Vector3f::new(r64(4.0), r64(16.0), r64(8.0)));
    }

    #[test]
    fn vector3_div() {
        let v1 = Vector3i::new(2, 8, 4);
        let v2 = Vector3f::new(r64(2.0), r64(8.0), r64(4.0));

        assert_eq!(v1 / 2, Vector3i::new(1, 4, 2));
        assert_eq!(v2 / r64(2.0), Vector3f::new(r64(1.0), r64(4.0), r64(2.0)));
    }

    #[test]
    fn vector3_div_assign() {
        let mut v1 = Vector3i::new(2, 8, 4);
        v1 /= 2;

        let mut v2 = Vector3f::new(r64(2.0), r64(8.0), r64(4.0));
        v2 /= r64(2.0);

        assert_eq!(v1, Vector3i::new(1, 4, 2));
        assert_eq!(v2, Vector3f::new(r64(1.0), r64(4.0), r64(2.0)));
    }

    #[test]
    fn vector3_neg() {
        let v1 = Vector3i::new(-1, 2, -3);
        let v2 = Vector3f::new(r64(-1.0), r64(2.0), r64(-3.0));

        assert_eq!(-v1, Vector3i::new(1, -2, 3));
        assert_eq!(-v2, Vector3f::new(r64(1.0), r64(-2.0), r64(3.0)));
    }

    #[test]
    fn vector3_length_squared() {
        let v1 = Vector3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(v1.length_squared(), r64(14.0));
    }

    #[test]
    fn vector3_length() {
        let v1 = Vector3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(v1.length(), r64(14.0).sqrt());
    }

    #[test]
    fn vector3_normalise() {
        let v1 = Vector3f::new(r64(2.0), r64(2.0), r64(2.0));
        assert_eq!(
            v1.normalise(),
            Vector3f::new(
                r64(1.0 / 3.0.sqrt()),
                r64(1.0 / 3.0.sqrt()),
                r64(1.0 / 3.0.sqrt())
            )
        );
    }

    #[test]
    fn vector3_dot() {
        let v1 = Vector3i::new(-1, 2, -3);
        let v2 = Vector3i::new(4, -5, 6);

        assert_eq!(v1.dot(v2), -32);
        assert_eq!(v1.dot_abs(v2), 32);
    }

    #[test]
    fn vector3_dot_normal3() {
        let v1 = Vector3i::new(-1, 2, -3);
        let n1 = Normal3i::new(4, -5, 6);

        assert_eq!(v1.dot(n1), -32);
        assert_eq!(v1.dot_abs(n1), 32);
    }

    #[test]
    fn vector3_cross() {
        let v1 = Vector3i::new(1, 2, 3);
        let v2 = Vector3i::new(4, 5, 6);

        assert_eq!(v1.cross(v2), Vector3::new(-3, 6, -3));
    }

    #[test]
    fn vector3_cross_normal3() {
        let v1 = Vector3i::new(1, 2, 3);
        let n1 = Normal3i::new(4, 5, 6);

        assert_eq!(v1.cross(n1), Vector3::new(-3, 6, -3));
    }

    #[test]
    fn vector3_face_forward() {
        let v1 = Vector3i::new(-1, 2, -3);
        let v2 = Vector3i::new(4, -5, 6);
        let v3 = Vector3i::new(-4, 5, -6);

        assert_eq!(v1.face_forward(v2), -v1);
        assert_eq!(v1.face_forward(v3), v1);
    }

    #[test]
    fn vector3_face_forward_normal3() {
        let v1 = Vector3i::new(-1, 2, -3);
        let n1 = Normal3i::new(4, -5, 6);
        let n2 = Normal3i::new(-4, 5, -6);

        assert_eq!(v1.face_forward(n1), -v1);
        assert_eq!(v1.face_forward(n2), v1);
    }

    #[test]
    fn vector3_coordinate_system() {
        let v1 = Vector3f::new(r64(1.0), r64(0.0), r64(0.0));
        let v2 = Vector3f::new(r64(0.0), r64(1.0), r64(0.0));

        assert_eq!(
            v1.coordinate_system(),
            (
                Vector3f::new(r64(1.0), r64(0.0), r64(0.0)),
                Vector3f::new(r64(0.0), r64(0.0), r64(1.0)),
                Vector3f::new(r64(0.0), r64(-1.0), r64(0.0)),
            )
        );
        assert_eq!(
            v2.coordinate_system(),
            (
                Vector3f::new(r64(0.0), r64(1.0), r64(0.0)),
                Vector3f::new(r64(0.0), r64(0.0), r64(-1.0)),
                Vector3f::new(r64(-1.0), r64(0.0), r64(0.0)),
            )
        );
    }
}
