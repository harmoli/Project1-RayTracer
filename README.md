Project 1 : GPU Ray Tracer
============================

 Overview : 
-----------------------------

For this project we have implemented a per-pixel parallel GPU ray tracer.  The ray tracer implements the following: 
1. Diffuse Shading
2. Phong shading
3. Soft shadows and area lights
4. Depth of field
5. Interactive Camera
6. Specular reflection
7. Raytraced shadows
8. Sphere surface point sampling
9. Multiple light system support
10. Cube intersection testing
11. Raycasting from a camera into a scene through a pixel grid

NOTES:

Depth of field :
Currently the depth of field is set to a focal length of 10.0.  This can be changed in the raytracer kernel by 
setting the focal length. The aperture can also be changed by changed the multiplicative factor of the ray origin jitter.

Interactive Camera :
The keystrokes to interact with the camera are as follows:

W : +y direction
X : -y direction
A : -x direction
D : +x direction
S : -z direction
Z : +z direction

Currently, we only allow for the user to interact with the camera by moving the camera's eye position.  
We plan on adding ways to interact with focal length, aperture and rotation of the camera in the near future.


[ SEE RENDERS FOLDER FOR RENDERED SCENES FROM RAYTRACER ]


Performance Analysis :
---------------------------

The trend for GPU has been motivated by performance optimization. For this ray tracer, we have added a 
framerate counter and use frames per second (fps) as a performance metric.  We will describe the following 
experiments : (1) moving lights outside of the geometry array, and(2) changing tile size.

### (1) Moving Lights Outside of the Geometry Array 
NOTE: This is no longer part of the code on the repo.
Originally, we had the lights as part of the geometry array, and having a secondary check. In order to 
minimize the number eof branches, we moved the lights outside the geometry array and had another light array 
that contained the light geometry (if the emmittance > 0).  When moving the light array that contained the light 
geometry to another array we had the following results:

(Ray Trace Depth = 1)
Without moving lights to another array : ~15 fps
Moving Light to Another Array : ~50 fps

Unfortunately, this code had issues with the shadow feelers, and so, we decided to move to another way of 
reducing the number of lights through which to traverse.  However, because we fell back to a way taht has 
a similar run time to naive checking of geometry, so the performance did not change.  

Perceivably, moving the light geometry into another array would help performance because there would be 
less checks, on a whole.  Not only would it decrease the number of objects in which to check while accumulating 
color on an intersection, but would also decrease the number of conditional branches that result from checking 
if the geometry is a light object or not.


### (2) Experiments in Tile Size 
Tile size ultimately affects the number of threads and warps used by the GPU.  As the hardware has physical 
limitations and tradeoffs, we decided to experiment with the affect of tile size on performance, and, as a 
result, the framerate.

Results:

____________________________
Tile Size || FPS

    4     || 5.7 / 193
    8     || 15.0 / 68
    16    || 15.0 / 72
    32    || 12.0 -13.5 / 85
    64    || Kernel error
    
(Ray Trace Depth = 8) 
3 reflective spheres, 1 light, all walls diffuse
____________________________
Tile Size || FPS / time (ms) 

    4     || 2.7 / 382 
    8     || 8.0 / 130
    16    || 7.7 / 134
    32    || 7.1 / 147
    64    || kernel error

(Ray Trace Depth = 8)
3 reflective spheres, 1 lights, all walls reflective
____________________________

The kernel error on a tile size of 64 makes sense because the warp size on our GPU is 32. However, it is 
apparent that the tile size, and, consequently, the number of threads run per block has an affect on the 
performance as a whole. As we can see, there is a maximal performance around a tile size of 8-16 (totaling 64-
256 threads per block). This coordinates with the half-warp size and breaking up of the threads amongst 
the blocks.  With tile size of 4, resources are not adequately split up, which causes it to slow, whereas a 
tile size of 32 splits up resources in an ineffective manner.


Future Optimizations :
---------------------------

This version of the code is, by no means, the best in terms of optimizations.  hHere we list some planned 
optimizations of the code that could drastically improve the performance.

### (1) Enforcing invariant in object parsing 
Currently, we have conditional branch that checks for the type of geometry and chooses the intersection 
test to use.  Since GPU performance is greatly affected by misprediction penalty, we can amortize this 
cost by preprocessing the cubes and spheres to be in separate arrays and checking them in order.  This way, 
we can remove this branch.

### (2) Ray Parallelization 
Currently, we are parallelizing based off of pixels.  This causes many wasted cycles when doing 
calculations such as reflection and refraction.  We can prevent this by changing the code to better 
maximize its usage of the GPU.  Instead of hhaving the CPU launch per pixel kernels for the image, we 
can copy rays that need to be bounced into a vector.  Then, we can write a scheduler on the CPU side that 
will launch kernels for these rays.  This would minimize the number of the wasted cycles.  However, it 
becomes a question of how to properly optimize this scheduler to take advantage of this scheduling scheme. 

### (3) Caching and selective update 
Since we are rendering a static scene, it is possible to cache data to prevent the necessity and time of 
creating and copying memory.  Similarly, we can selectively update the render based on changes to the scene.

### (4) Splitting up the render by tile
Instead of rendering the entire picture at once, we can use cache locality to our advantage by tiling the 
rendered picture and rendering each tile successively. By pulling in memory that is close to each other, 
we cut down on the time need to fix and deal with loading memory that is farther away on cache misses on 
the CPU before sending the information to the GPU for rendering.  

Acknowledgements :
--------------------------------

Many thanks to Karl Li for providing the base code for this assignment.  The framerate calculator was 
borrowed from previous CIS 563 code, and was provided by Eric Chan.
