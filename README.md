# mgmath
```mgmath``` is a small vector/matrix math library with some basic transform support, and is meant to be used mainly in 3D graphics programming.
It is a header-only library in the most pure sense, as it only consists of a single header file, which contains everything

## Examples
Some usage examples for vectors and matrices
### Vectors
- To create a basic vector with a number of components:
  - `vec<2, TYPE>` with `x` `y` (short `vec2...`)
  - `vec<3, TYPE>` with `x` `y` `z` (short `vec3...`)
  - `vec<4, TYPE>` with `x` `y` `z` `w` (short `vec4...`)
- After vector name `vecx` you can add `i`, `u`, `f`, `d`:
  - `i` for an intiger vector (`vec2i32` for int32_t vector)
  - `u` for an unsigned intiger vector (`vec2u32` for uint32_t vector)
  - `f` for a float vector (`vec2f` for float vector)
  - `d` for a double vector (`vec2d` for double vector)
- You can also use anything from 8-bit intigers, to 64-bit intigers (signed and unsigned)
  - `vec2u8` `vec2u16` `vec2u32` `vec2u64`
- You can also use the `vec` template to create your own vector
  - `vec<4, float>` is the equivalent of `vec4f`
  - `vec<3, uint16_t>` is the equivalent of `vec3u16`
- The `vec` template contains the members `x` `y` `z` `w`, equivalent to getting an element using `[0]` `[1]` `[2]` `[3]`, if the vector has a size of `2` `3` or `4`
  - `vec<2, float> v;` - `v[0] = 10;` is the same as `v.x = 10;`

### Matrices
- To create a basic matrix, it's similar to vector
- The matrices already declared in the header are only square; To use a non-square matrix, use the template:
  - `mat<3, 3, float>` is the equivalent of mat3f
  - `mat<4, 3, float>` creates a 4x3 matrix of floats

### Transforms
- Matrices have functions to rotate along any axis in any given order:
  - `rotate2d` rotates the matrix, then returns a reference to it
  - `rotated2d` returns a rotated version of the matrix, without modifying the original
  - `rotate3d_xyz` rotated the matrix in the order `x`, then `y`, then `z`
  - And so on and so forth
- 2D matrices:
  - 2x2 for `rotation` and `scale`
  - 3x3 for `position`, `rotation`, `scale` and `skew`
- 3D matrices:
  - 3x3 for `rotation` and `scale` in all 3 axis
  - 4x4 for `position`, `rotation`, `scale` and `skew`

### Quaternions
- Quaternions are used to handle rotations in a more optimized, and easier way than with rotation matrices:
  - `quatf` and `quatd` are available and contain utility functions for rotating vectors
  - First, a `quat` must be generated using the static `from_angle` or `from_angle_safe` function in the `quat` class
  - Next, that `quat` can be used to rotate a vector, or can be multiplied with another `quat` to combine the 2 rotations
  - `as_rotation_mat3` and `as_rotation_mat4` are also available to generation rotation matrices from a quaternion

### Extra
- Everything is tightly packed, so a list of float vectors is the same as a larger list of floats
  - This means you can easily send them to OpenGL, Vulkan or other APIs that require you to send data in large packs
- There is SIMD support on `x86_64` and `amd64` thanks to SSE and AVX. (so far only for `vec4f`, and `mat4f`)

### To Do
- [ ] Add remaining transform functions for matrices
- [ ] Fix SIMD problems on AMD when compiled with MSVC
- [ ] Add SIMD support for every possible vector
- [ ] Add SIMD support on arm
