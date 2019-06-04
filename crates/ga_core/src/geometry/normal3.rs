use crate::constants::Float;

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