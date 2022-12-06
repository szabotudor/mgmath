# mgmath
```mgmath``` is a small vector/matrix math library with some basic transform support, and is meant to be used mainly in 3D graphics programming.
It is a header-only library in the most pure sense, as it only consists of a single header file, which contains everything

## Heads Up
This is a work in progress, so don't shy away from contributing to it, or telling me how to improve it.

## Examples
Some usage examples for vectors and matrices
### Vectors
- To create a basic vector with a number of components:
  - ```vec2``` with x and y
  - ```vec3``` with x, y, and z
  - ```vec4``` with x, y, z, and w
- Preface the vector with:
  - ```i``` for an intiger vector (```ivec2```)
  - ```d``` for a double vector (```dvec2```)
  - ```ui``` for an unsigned intiger vector (```uivec2```)
- You can also add a number between the preface and "vec" to use anything from 8-bit intigers, to 64-bit intigers (signed and unsigned)
  - ```i32vec3``` ```ui8vec4``` ```ui64vec2``` are a few examples
- You can also use the ```vec``` template to create your own vector with up to 255 components
  - ```vec<4, float>``` is the equivalent of ```vec4```
  - ```vec<3, uint16_t>``` is the equivalent of ```ui16vec3```
- The ```vec``` template doesn't contain the xyzw components, and to get/set an element use the ```[]``` operator
  - ```vec<2, float> v; v[0] = 10; v[1] = 14;```

### Matrices
- To create a basic matrix, it's similar to vector
- The matrices already declared in the header are only square; To use a non-square matrix, use the template:
  - ```mat<4, 3, float>``` creates a 4x3 matrix of floats

### Transforms
- So far transforms are not 100%, I am still working on them, but they can still be used as they stand
- The function names are pretty self-explanetory
- Quick thing to note: ```rotate``` rotates the transform globaly, and ```rotateLook``` rotates it localy, and is recommended when used as the transform for a camera in 3D.
