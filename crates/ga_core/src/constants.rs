//! Global constants and type definitions that can be used across Ganymede.

/// An alias for the [`f32`](https://doc.rust-lang.org/std/f32/index.html) type.
///
/// Throughout Ganymede, all references to floating point values should use this
/// alias, making it possible to switch to double precision values in the future
/// if necessary with minimal fuss.
pub type Float = f32;
