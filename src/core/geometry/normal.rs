//! Normal representations for two and three dimensions.

use std::cmp::{max, min};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use noisy_float::prelude::*;
use num::{abs, Float, Signed, Zero};

use super::ops::{Cross, Dot, FaceForward, Length, Normalise};
use super::vector::{Vector2, Vector3};
use super::{Prim, PrimFloat, PrimItem, PrimSigned};

/// Two-dimensional normal of integers.
pub type Normal2i = Normal2<isize>;
/// Two-dimensional normal of floats.
pub type Normal2f = Normal2<R64>;
/// Three-dimensional normal of integers.
pub type Normal3i = Normal3<isize>;
/// Three-dimensional normal of floats.
pub type Normal3f = Normal3<R64>;

/// A generic, two-dimensional normal.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct Normal2<T: PrimItem> {
    pub x: T,
    pub y: T,
}

// Prim trait implementation

impl<T: PrimItem> Prim for Normal2<T> {
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
        Normal2 {
            x: min(self.x, other.x),
            y: min(self.y, other.y),
        }
    }

    fn max(self, other: Self) -> Self {
        Normal2 {
            x: max(self.x, other.x),
            y: max(self.y, other.y),
        }
    }
}

// PrimSigned trait implementation

impl<T: PrimItem + Signed> PrimSigned for Normal2<T> {
    fn abs(self) -> Self {
        Normal2 {
            x: abs(self.x),
            y: abs(self.y),
        }
    }
}

// PrimFloat trait implementation

impl<T: PrimItem + Float> PrimFloat for Normal2<T> {
    fn floor(self) -> Self {
        Normal2 {
            x: self.x.floor(),
            y: self.y.floor(),
        }
    }

    fn ceil(self) -> Self {
        Normal2 {
            x: self.x.ceil(),
            y: self.y.ceil(),
        }
    }
}

// Constructors

impl<T: PrimItem> Normal2<T> {
    /// Construct a new Normal2 from individual component values.
    pub fn new(x: T, y: T) -> Self {
        Normal2 { x, y }
    }

    /// Construct a Vector2 from a Normal2.
    pub fn to_vector(self) -> Vector2<T> {
        Vector2 {
            x: self.x,
            y: self.y,
        }
    }
}

// Indexing traits

impl<T: PrimItem> Index<usize> for Normal2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index out of bounds (Normal2)"),
        }
    }
}

impl<T: PrimItem> IndexMut<usize> for Normal2<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index out of bounds (Normal2)"),
        }
    }
}

impl<T: PrimItem> Normal2<T> {
    pub fn permute(&self, x: usize, y: usize) -> Normal2<T> {
        Normal2 {
            x: self[x],
            y: self[y],
        }
    }
}

// Operator traits

impl<T: PrimItem + Add<Output = T>> Add for Normal2<T> {
    type Output = Normal2<T>;

    fn add(self, other: Normal2<T>) -> Self::Output {
        Normal2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl<T: PrimItem + AddAssign> AddAssign for Normal2<T> {
    fn add_assign(&mut self, other: Normal2<T>) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl<T: PrimItem + Sub<Output = T>> Sub for Normal2<T> {
    type Output = Normal2<T>;

    fn sub(self, other: Normal2<T>) -> Self::Output {
        Normal2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl<T: PrimItem + SubAssign> SubAssign for Normal2<T> {
    fn sub_assign(&mut self, other: Normal2<T>) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl<T: PrimItem + Mul<T, Output = T>> Mul<T> for Normal2<T> {
    type Output = Normal2<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Normal2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T: PrimItem + MulAssign> MulAssign<T> for Normal2<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl<T: PrimItem + Div<T, Output = T>> Div<T> for Normal2<T> {
    type Output = Normal2<T>;

    fn div(self, rhs: T) -> Self::Output {
        Normal2 {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl<T: PrimItem + DivAssign> DivAssign<T> for Normal2<T> {
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl<T: PrimItem + Signed> Neg for Normal2<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Normal2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

// Length operation

impl<T: PrimItem + Float> Length for Normal2<T> {
    fn length_squared(self) -> Self::Item {
        self.x * self.x + self.y * self.y
    }
}

// Normalise operation

impl<T: PrimItem + Float> Normalise for Normal2<T> {
    fn normalise(self) -> Self {
        self / self.length()
    }
}

// Dot operation

impl<T: PrimItem + Signed> Dot for Normal2<T> {
    fn dot(self, rhs: Self) -> Self::Item {
        self.x * rhs.x + self.y * rhs.y
    }
}

impl<T: PrimItem + Signed> Dot<Vector2<T>> for Normal2<T> {
    fn dot(self, rhs: Vector2<T>) -> Self::Item {
        self.x * rhs.x + self.y * rhs.y
    }
}

/// A generic, three-dimensional normal.
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct Normal3<T: PrimItem> {
    pub x: T,
    pub y: T,
    pub z: T,
}

// Prim trait implementation

impl<T: PrimItem> Prim for Normal3<T> {
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
        Normal3 {
            x: min(self.x, other.x),
            y: min(self.y, other.y),
            z: min(self.z, other.z),
        }
    }

    fn max(self, other: Self) -> Self {
        Normal3 {
            x: max(self.x, other.x),
            y: max(self.y, other.y),
            z: max(self.z, other.z),
        }
    }
}

// PrimSigned trait implementation

impl<T: PrimItem + Signed> PrimSigned for Normal3<T> {
    fn abs(self) -> Self {
        Normal3 {
            x: abs(self.x),
            y: abs(self.y),
            z: abs(self.z),
        }
    }
}

// PrimFloat trait implementation

impl<T: PrimItem + Float> PrimFloat for Normal3<T> {
    fn floor(self) -> Self {
        Normal3 {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
        }
    }

    fn ceil(self) -> Self {
        Normal3 {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
        }
    }
}

// Constructors

impl<T: PrimItem> Normal3<T> {
    /// Construct a new Normal3 from individual component values.
    pub fn new(x: T, y: T, z: T) -> Self {
        Normal3 { x, y, z }
    }

    /// Construct a new Vector3 from a Normal3.
    pub fn to_vector(self) -> Vector3<T> {
        Vector3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

// Indexing traits

impl<T: PrimItem> Index<usize> for Normal3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index out of bounds (Normal3)"),
        }
    }
}

impl<T: PrimItem> IndexMut<usize> for Normal3<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index out of bounds (Normal3)"),
        }
    }
}

impl<T: PrimItem> Normal3<T> {
    pub fn permute(&self, x: usize, y: usize, z: usize) -> Normal3<T> {
        Normal3 {
            x: self[x],
            y: self[y],
            z: self[z],
        }
    }
}

// Operator traits

impl<T: PrimItem + Add<Output = T>> Add for Normal3<T> {
    type Output = Normal3<T>;

    fn add(self, other: Normal3<T>) -> Self::Output {
        Normal3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: PrimItem + AddAssign> AddAssign for Normal3<T> {
    fn add_assign(&mut self, other: Normal3<T>) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<T: PrimItem + Sub<Output = T>> Sub for Normal3<T> {
    type Output = Normal3<T>;

    fn sub(self, other: Normal3<T>) -> Self::Output {
        Normal3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: PrimItem + SubAssign> SubAssign for Normal3<T> {
    fn sub_assign(&mut self, other: Normal3<T>) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl<T: PrimItem + Mul<T, Output = T>> Mul<T> for Normal3<T> {
    type Output = Normal3<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Normal3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T: PrimItem + MulAssign> MulAssign<T> for Normal3<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl<T: PrimItem + Div<T, Output = T>> Div<T> for Normal3<T> {
    type Output = Normal3<T>;

    fn div(self, rhs: T) -> Self::Output {
        Normal3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T: PrimItem + DivAssign> DivAssign<T> for Normal3<T> {
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl<T: PrimItem + Signed> Neg for Normal3<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Normal3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// Length operation

impl<T: PrimItem + Float> Length for Normal3<T> {
    fn length_squared(self) -> Self::Item {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}

// Normalise operation

impl<T: PrimItem + Float> Normalise for Normal3<T> {
    fn normalise(self) -> Self {
        self / self.length()
    }
}

// Dot operation

impl<T: PrimItem + Signed> Dot for Normal3<T> {
    fn dot(self, rhs: Self) -> Self::Item {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl<T: PrimItem + Signed> Dot<Vector3<T>> for Normal3<T> {
    fn dot(self, rhs: Vector3<T>) -> Self::Item {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

// Cross operation

impl<T: PrimItem + Signed> Cross<Vector3<T>> for Normal3<T> {
    type Output = Self;

    fn cross(self, rhs: Vector3<T>) -> Self::Output {
        Normal3::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
}

// Face-forward operation

impl<T: PrimItem + Signed> FaceForward for Normal3<T> {
    type Output = Self;

    fn face_forward(self, rhs: Self) -> Self::Output {
        match self.dot(rhs) < Zero::zero() {
            true => -self,
            false => self,
        }
    }
}

impl<T: PrimItem + Signed> FaceForward<Vector3<T>> for Normal3<T> {
    type Output = Self;

    fn face_forward(self, rhs: Vector3<T>) -> Self::Output {
        match self.dot(rhs) < Zero::zero() {
            true => -self,
            false => self,
        }
    }
}

// Tests

#[cfg(test)]
#[cfg_attr(tarpaulin, skip)]
mod tests {
    use super::super::vector::{Vector2i, Vector3i};
    use super::*;

    // Normal2 tests

    #[test]
    fn normal2_to_vector2() {
        let n1 = Normal2i::new(1, 2);

        assert_eq!(n1.to_vector(), Vector2i::new(1, 2));
    }

    #[test]
    fn normal2_min_component() {
        let n1 = Normal2i::new(1, 2);
        let n2 = Normal2f::new(r64(1.0), r64(2.0));

        assert_eq!(n1.min_component(), 1);
        assert_eq!(n2.min_component(), r64(1.0));
    }

    #[test]
    fn normal2_max_component() {
        let n1 = Normal2i::new(1, 2);
        let n2 = Normal2f::new(r64(1.0), r64(2.0));

        assert_eq!(n1.max_component(), 2);
        assert_eq!(n2.max_component(), r64(2.0));
    }

    #[test]
    fn normal2_min_dimension() {
        let n1 = Normal2i::new(1, 2);
        let n2 = Normal2f::new(r64(1.0), r64(2.0));

        assert_eq!(n1.min_dimension(), 0);
        assert_eq!(n2.min_dimension(), 0);
    }

    #[test]
    fn normal2_max_dimension() {
        let n1 = Normal2i::new(1, 2);
        let n2 = Normal2f::new(r64(1.0), r64(2.0));

        assert_eq!(n1.max_dimension(), 1);
        assert_eq!(n2.max_dimension(), 1);
    }

    #[test]
    fn normal2_min() {
        let n1 = Normal2i::new(2, 8);
        let n2 = Normal2i::new(3, 5);
        let n3 = Normal2f::new(r64(2.0), r64(8.0));
        let n4 = Normal2f::new(r64(3.0), r64(5.0));

        assert_eq!(n1.min(n2), Normal2i::new(2, 5));
        assert_eq!(n3.min(n4), Normal2f::new(r64(2.0), r64(5.0)));
    }

    #[test]
    fn normal2_max() {
        let n1 = Normal2i::new(2, 8);
        let n2 = Normal2i::new(3, 5);
        let n3 = Normal2f::new(r64(2.0), r64(8.0));
        let n4 = Normal2f::new(r64(3.0), r64(5.0));

        assert_eq!(n1.max(n2), Normal2i::new(3, 8));
        assert_eq!(n3.max(n4), Normal2f::new(r64(3.0), r64(8.0)));
    }

    #[test]
    fn normal2_abs() {
        let n1 = Normal2i::new(-1, 2);
        let n2 = Normal2f::new(r64(-1.0), r64(2.0));

        assert_eq!(n1.abs(), Normal2i::new(1, 2));
        assert_eq!(n2.abs(), Normal2f::new(r64(1.0), r64(2.0)));
    }

    #[test]
    fn normal2_floor() {
        let n1 = Normal2f::new(r64(1.1), r64(4.2));

        assert_eq!(n1.floor(), Normal2f::new(r64(1.0), r64(4.0)));
    }

    #[test]
    fn normal2_ceil() {
        let n1 = Normal2f::new(r64(1.1), r64(4.2));

        assert_eq!(n1.ceil(), Normal2f::new(r64(2.0), r64(5.0)));
    }

    #[test]
    fn normal2_index() {
        let n1 = Normal2i::new(0, 1);

        assert_eq!(n1[0], 0);
        assert_eq!(n1[1], 1);
    }

    #[test]
    fn normal2_index_mut() {
        let mut n1 = Normal2i::new(0, 1);
        n1[0] = 3;
        n1[1] = 4;

        assert_eq!(n1[0], 3);
        assert_eq!(n1[1], 4);
    }

    #[test]
    #[should_panic]
    fn normal2_index_out_of_bounds() {
        let _ = Normal2i::new(0, 1)[2];
    }

    #[test]
    #[should_panic]
    fn normal2_index_mut_out_of_bounds() {
        let mut n1 = Normal2i::new(0, 1);
        n1[2] = 2;
    }

    #[test]
    fn normal2_permute() {
        let n1 = Normal2i::new(1, 2);

        assert_eq!(n1.permute(1, 0), Normal2i::new(2, 1));
    }

    #[test]
    fn normal2_add() {
        let n1 = Normal2i::new(2, 8);
        let n2 = Normal2i::new(3, 5);
        let n3 = Normal2f::new(r64(2.0), r64(8.0));
        let n4 = Normal2f::new(r64(3.0), r64(5.0));

        assert_eq!(n1 + n2, Normal2i::new(5, 13));
        assert_eq!(n3 + n4, Normal2f::new(r64(5.0), r64(13.0)));
    }

    #[test]
    fn normal2_add_assign() {
        let mut n1 = Normal2i::new(2, 8);
        let n2 = Normal2i::new(3, 5);
        n1 += n2;

        let mut n3 = Normal2f::new(r64(2.0), r64(8.0));
        let n4 = Normal2f::new(r64(3.0), r64(5.0));
        n3 += n4;

        assert_eq!(n1, Normal2i::new(5, 13));
        assert_eq!(n3, Normal2f::new(r64(5.0), r64(13.0)));
    }

    #[test]
    fn normal2_sub() {
        let n1 = Normal2i::new(2, 8);
        let n2 = Normal2i::new(3, 5);
        let n3 = Normal2f::new(r64(2.0), r64(8.0));
        let n4 = Normal2f::new(r64(3.0), r64(5.0));

        assert_eq!(n1 - n2, Normal2i::new(-1, 3));
        assert_eq!(n3 - n4, Normal2f::new(r64(-1.0), r64(3.0)));
    }

    #[test]
    fn normal2_sub_assign() {
        let mut n1 = Normal2i::new(2, 8);
        let n2 = Normal2i::new(3, 5);
        n1 -= n2;

        let mut n3 = Normal2f::new(r64(2.0), r64(8.0));
        let n4 = Normal2f::new(r64(3.0), r64(5.0));
        n3 -= n4;

        assert_eq!(n1, Normal2i::new(-1, 3));
        assert_eq!(n3, Normal2f::new(r64(-1.0), r64(3.0)));
    }

    #[test]
    fn normal2_mul() {
        let n1 = Normal2i::new(2, 8);
        let n2 = Normal2f::new(r64(2.0), r64(8.0));

        assert_eq!(n1 * 2, Normal2i::new(4, 16));
        assert_eq!(n2 * r64(2.0), Normal2f::new(r64(4.0), r64(16.0)));
    }

    #[test]
    fn normal2_mul_assign() {
        let mut n1 = Normal2i::new(2, 8);
        n1 *= 2;

        let mut n2 = Normal2f::new(r64(2.0), r64(8.0));
        n2 *= r64(2.0);

        assert_eq!(n1, Normal2i::new(4, 16));
        assert_eq!(n2, Normal2f::new(r64(4.0), r64(16.0)));
    }

    #[test]
    fn normal2_div() {
        let n1 = Normal2i::new(2, 8);
        let n2 = Normal2f::new(r64(2.0), r64(8.0));

        assert_eq!(n1 / 2, Normal2i::new(1, 4));
        assert_eq!(n2 / r64(2.0), Normal2f::new(r64(1.0), r64(4.0)));
    }

    #[test]
    fn normal2_div_assign() {
        let mut n1 = Normal2i::new(2, 8);
        n1 /= 2;

        let mut n2 = Normal2f::new(r64(2.0), r64(8.0));
        n2 /= r64(2.0);

        assert_eq!(n1, Normal2i::new(1, 4));
        assert_eq!(n2, Normal2f::new(r64(1.0), r64(4.0)));
    }

    #[test]
    fn normal2_neg() {
        let n1 = Normal2i::new(-1, 2);
        let n2 = Normal2f::new(r64(-1.0), r64(2.0));

        assert_eq!(-n1, Normal2i::new(1, -2));
        assert_eq!(-n2, Normal2f::new(r64(1.0), r64(-2.0)));
    }

    #[test]
    fn normal2_length_squared() {
        let n1 = Normal2f::new(r64(1.0), r64(2.0));

        assert_eq!(n1.length_squared(), r64(5.0));
    }

    #[test]
    fn normal2_length() {
        let n1 = Normal2f::new(r64(1.0), r64(2.0));

        assert_eq!(n1.length(), r64(5.0).sqrt());
    }

    #[test]
    fn normal2_normalise() {
        let n1 = Normal2f::new(r64(2.0), r64(2.0));
        assert_eq!(
            n1.normalise(),
            Normal2f::new(r64(1.0 / 2.0.sqrt()), r64(1.0 / 2.0.sqrt()))
        );
    }

    #[test]
    fn normal2_dot() {
        let n1 = Normal2i::new(-1, 2);
        let n2 = Normal2i::new(3, -4);

        assert_eq!(n1.dot(n2), -11);
        assert_eq!(n1.dot_abs(n2), 11);
    }

    #[test]
    fn normal2_dot_vector2() {
        let n1 = Normal2i::new(-1, 2);
        let v1 = Vector2i::new(3, -4);

        assert_eq!(n1.dot(v1), -11);
        assert_eq!(n1.dot_abs(v1), 11);
    }

    // Normal3 tests

    #[test]
    fn normal3_to_vector3() {
        let n1 = Normal3i::new(1, 2, 3);

        assert_eq!(n1.to_vector(), Vector3i::new(1, 2, 3));
    }

    #[test]
    fn normal3_min_component() {
        let n1 = Normal3i::new(1, 2, 3);
        let n2 = Normal3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(n1.min_component(), 1);
        assert_eq!(n2.min_component(), r64(1.0));
    }

    #[test]
    fn normal3_max_component() {
        let n1 = Normal3i::new(1, 2, 3);
        let n2 = Normal3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(n1.max_component(), 3);
        assert_eq!(n2.max_component(), r64(3.0));
    }

    #[test]
    fn normal3_min_dimension() {
        let n1 = Normal3i::new(1, 2, 3);
        let n2 = Normal3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(n1.min_dimension(), 0);
        assert_eq!(n2.min_dimension(), 0);
    }

    #[test]
    fn normal3_max_dimension() {
        let n1 = Normal3i::new(1, 2, 3);
        let n2 = Normal3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(n1.max_dimension(), 2);
        assert_eq!(n2.max_dimension(), 2);
    }

    #[test]
    fn normal3_min() {
        let n1 = Normal3i::new(2, 8, 4);
        let n2 = Normal3i::new(3, 5, 6);
        let n3 = Normal3f::new(r64(2.0), r64(8.0), r64(4.0));
        let n4 = Normal3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(n1.min(n2), Normal3i::new(2, 5, 4));
        assert_eq!(n3.min(n4), Normal3f::new(r64(2.0), r64(5.0), r64(4.0)));
    }

    #[test]
    fn normal3_max() {
        let n1 = Normal3i::new(2, 8, 4);
        let n2 = Normal3i::new(3, 5, 6);
        let n3 = Normal3f::new(r64(2.0), r64(8.0), r64(4.0));
        let n4 = Normal3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(n1.max(n2), Normal3i::new(3, 8, 6));
        assert_eq!(n3.max(n4), Normal3f::new(r64(3.0), r64(8.0), r64(6.0)));
    }

    #[test]
    fn normal3_abs() {
        let n1 = Normal3i::new(-1, 2, -4);
        let n2 = Normal3f::new(r64(-1.0), r64(2.0), r64(-4.0));

        assert_eq!(n1.abs(), Normal3i::new(1, 2, 4));
        assert_eq!(n2.abs(), Normal3f::new(r64(1.0), r64(2.0), r64(4.0)));
    }

    #[test]
    fn normal3_floor() {
        let n1 = Normal3f::new(r64(1.1), r64(4.2), r64(-2.1));

        assert_eq!(n1.floor(), Normal3f::new(r64(1.0), r64(4.0), r64(-3.0)));
    }

    #[test]
    fn normal3_ceil() {
        let n1 = Normal3f::new(r64(1.1), r64(4.2), r64(-2.1));

        assert_eq!(n1.ceil(), Normal3f::new(r64(2.0), r64(5.0), r64(-2.0)));
    }

    #[test]
    fn normal3_index() {
        let n1 = Normal3i::new(0, 1, 2);

        assert_eq!(n1[0], 0);
        assert_eq!(n1[1], 1);
        assert_eq!(n1[2], 2);
    }

    #[test]
    fn normal3_index_mut() {
        let mut n1 = Normal3i::new(0, 1, 2);
        n1[0] = 3;
        n1[1] = 4;
        n1[2] = 5;

        assert_eq!(n1[0], 3);
        assert_eq!(n1[1], 4);
        assert_eq!(n1[2], 5);
    }

    #[test]
    #[should_panic]
    fn normal3_index_out_of_bounds() {
        let _ = Normal3i::new(0, 1, 2)[3];
    }

    #[test]
    #[should_panic]
    fn normal3_index_mut_out_of_bounds() {
        let mut n1 = Normal3i::new(0, 1, 2);
        n1[3] = 3;
    }

    #[test]
    fn normal3_permute() {
        let n1 = Normal3i::new(1, 2, 3);

        assert_eq!(n1.permute(2, 1, 0), Normal3i::new(3, 2, 1));
    }

    #[test]
    fn normal3_add() {
        let n1 = Normal3i::new(2, 8, 4);
        let n2 = Normal3i::new(3, 5, 6);
        let n3 = Normal3f::new(r64(2.0), r64(8.0), r64(4.0));
        let n4 = Normal3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(n1 + n2, Normal3i::new(5, 13, 10));
        assert_eq!(n3 + n4, Normal3f::new(r64(5.0), r64(13.0), r64(10.0)));
    }

    #[test]
    fn normal3_add_assign() {
        let mut n1 = Normal3i::new(2, 8, 4);
        let n2 = Normal3i::new(3, 5, 6);
        n1 += n2;

        let mut n3 = Normal3f::new(r64(2.0), r64(8.0), r64(4.0));
        let n4 = Normal3f::new(r64(3.0), r64(5.0), r64(6.0));
        n3 += n4;

        assert_eq!(n1, Normal3i::new(5, 13, 10));
        assert_eq!(n3, Normal3f::new(r64(5.0), r64(13.0), r64(10.0)));
    }

    #[test]
    fn normal3_sub() {
        let n1 = Normal3i::new(2, 8, 4);
        let n2 = Normal3i::new(3, 5, 6);
        let n3 = Normal3f::new(r64(2.0), r64(8.0), r64(4.0));
        let n4 = Normal3f::new(r64(3.0), r64(5.0), r64(6.0));

        assert_eq!(n1 - n2, Normal3i::new(-1, 3, -2));
        assert_eq!(n3 - n4, Normal3f::new(r64(-1.0), r64(3.0), r64(-2.0)));
    }

    #[test]
    fn normal3_sub_assign() {
        let mut n1 = Normal3i::new(2, 8, 4);
        let n2 = Normal3i::new(3, 5, 6);
        n1 -= n2;

        let mut n3 = Normal3f::new(r64(2.0), r64(8.0), r64(4.0));
        let n4 = Normal3f::new(r64(3.0), r64(5.0), r64(6.0));
        n3 -= n4;

        assert_eq!(n1, Normal3i::new(-1, 3, -2));
        assert_eq!(n3, Normal3f::new(r64(-1.0), r64(3.0), r64(-2.0)));
    }

    #[test]
    fn normal3_mul() {
        let n1 = Normal3i::new(2, 8, 4);
        let n2 = Normal3f::new(r64(2.0), r64(8.0), r64(4.0));

        assert_eq!(n1 * 2, Normal3i::new(4, 16, 8));
        assert_eq!(n2 * r64(2.0), Normal3f::new(r64(4.0), r64(16.0), r64(8.0)));
    }

    #[test]
    fn normal3_mul_assign() {
        let mut n1 = Normal3i::new(2, 8, 4);
        n1 *= 2;

        let mut n2 = Normal3f::new(r64(2.0), r64(8.0), r64(4.0));
        n2 *= r64(2.0);

        assert_eq!(n1, Normal3i::new(4, 16, 8));
        assert_eq!(n2, Normal3f::new(r64(4.0), r64(16.0), r64(8.0)));
    }

    #[test]
    fn normal3_div() {
        let n1 = Normal3i::new(2, 8, 4);
        let n2 = Normal3f::new(r64(2.0), r64(8.0), r64(4.0));

        assert_eq!(n1 / 2, Normal3i::new(1, 4, 2));
        assert_eq!(n2 / r64(2.0), Normal3f::new(r64(1.0), r64(4.0), r64(2.0)));
    }

    #[test]
    fn normal3_div_assign() {
        let mut n1 = Normal3i::new(2, 8, 4);
        n1 /= 2;

        let mut n2 = Normal3f::new(r64(2.0), r64(8.0), r64(4.0));
        n2 /= r64(2.0);

        assert_eq!(n1, Normal3i::new(1, 4, 2));
        assert_eq!(n2, Normal3f::new(r64(1.0), r64(4.0), r64(2.0)));
    }

    #[test]
    fn normal3_neg() {
        let n1 = Normal3i::new(-1, 2, -3);
        let n2 = Normal3f::new(r64(-1.0), r64(2.0), r64(-3.0));

        assert_eq!(-n1, Normal3i::new(1, -2, 3));
        assert_eq!(-n2, Normal3f::new(r64(1.0), r64(-2.0), r64(3.0)));
    }

    #[test]
    fn normal3_length_squared() {
        let n1 = Normal3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(n1.length_squared(), r64(14.0));
    }

    #[test]
    fn normal3_length() {
        let n1 = Normal3f::new(r64(1.0), r64(2.0), r64(3.0));

        assert_eq!(n1.length(), r64(14.0).sqrt());
    }

    #[test]
    fn normal3_normalise() {
        let n1 = Normal3f::new(r64(2.0), r64(2.0), r64(2.0));
        assert_eq!(
            n1.normalise(),
            Normal3f::new(
                r64(1.0 / 3.0.sqrt()),
                r64(1.0 / 3.0.sqrt()),
                r64(1.0 / 3.0.sqrt())
            )
        );
    }

    #[test]
    fn normal3_dot() {
        let n1 = Normal3i::new(-1, 2, -3);
        let n2 = Normal3i::new(4, -5, 6);

        assert_eq!(n1.dot(n2), -32);
        assert_eq!(n1.dot_abs(n2), 32);
    }

    #[test]
    fn normal3_dot_vector3() {
        let n1 = Normal3i::new(-1, 2, -3);
        let v1 = Vector3i::new(4, -5, 6);

        assert_eq!(n1.dot(v1), -32);
        assert_eq!(n1.dot_abs(v1), 32);
    }

    #[test]
    fn normal3_cross_vector3() {
        let n1 = Normal3i::new(1, 2, 3);
        let v1 = Vector3i::new(4, 5, 6);

        assert_eq!(n1.cross(v1), Normal3::new(-3, 6, -3));
    }

    #[test]
    fn normal3_face_forward() {
        let n1 = Normal3i::new(-1, 2, -3);
        let n2 = Normal3i::new(4, -5, 6);
        let n3 = Normal3i::new(-4, 5, -6);

        assert_eq!(n1.face_forward(n2), -n1);
        assert_eq!(n1.face_forward(n3), n1);
    }

    #[test]
    fn normal3_face_forward_vector3() {
        let n1 = Normal3i::new(-1, 2, -3);
        let v1 = Vector3i::new(4, -5, 6);
        let v2 = Vector3i::new(-4, 5, -6);

        assert_eq!(n1.face_forward(v1), -n1);
        assert_eq!(n1.face_forward(v2), n1);
    }
}
