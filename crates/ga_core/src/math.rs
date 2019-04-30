use std::cmp::PartialOrd;
use std::ops::{Add, Div, Mul, Sub};

use num::{PrimInt, Zero};

/// Clamps the given value `val` between the `low` and `high`.
pub fn clamp<T>(val: T, low: T, high: T) -> T
where
    T: PartialOrd,
{
    if val < low {
        low
    } else if val > high {
        high
    } else {
        val
    }
}

/// Modulus that returns a positive value for a negative numbers.
pub fn modulus<T>(a: T, b: T) -> T
where
    T: Copy
        + PartialOrd
        + Zero
        + PrimInt
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    let result = a - (a / b) * b;
    if result < Zero::zero() {
        result + b
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    // Clamp

    #[test]
    fn fn_clamp() {
        let given: f32 = clamp(2.0, 1.0, 3.0);
        let expected: f32 = 2.0;
        assert_approx_eq!(given, expected);

        let given: f32 = clamp(1.0, 2.0, 3.0);
        let expected: f32 = 2.0;
        assert_approx_eq!(given, expected);

        let given: f32 = clamp(3.0, 1.0, 2.0);
        let expected: f32 = 2.0;
        assert_approx_eq!(given, expected);
    }

    // Modulus

    #[test]
    fn fn_modulus() {
        let given: i32 = modulus(24, 13);
        let expected: i32 = 11;
        assert_eq!(given, expected);

        let given: i32 = modulus(24, -13);
        let expected: i32 = 11;
        assert_eq!(given, expected);

        let given: i32 = modulus(-24, 13);
        let expected: i32 = 2;
        assert_eq!(given, expected);
    }
}
