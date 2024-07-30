# Web3D Lightmapper

This is project allows you to create light maps for BabylonJS meshes within the browser. The system is built using C++ and WebAssembly using the Emscripten toolchain. It works by first UV unwrapping each mesh and packing into a single atlas using the xatlas library. It then creates a bounding volume hierarchy 3D data structure from the whole scene using a cpp bvh library. The bounding volume hierarchy is packed linearly in depth first order in a memory efficient way, which makes it suitable for a linear stackless traversal on the GPU. To generate the actual light map a GPU based path tracer was written using WebGL2, which basically bounces light rays around the scene using the BVH to speed things up, bouncing off surfaces until a light is hit (this is simplified somewhat but basically true). All the scene data (like the BVH, materials, meshes etc.) is packed into GL textures which are used by a fragment shader that does the actual path tracing. The uv maps / atlases (with position and normal information too) is copied to a vertex buffer object for rendering (the first ray for each pixel starts from the interpolated position vertex and propagates along the interpolated normal etc.). There are a few simple filters including a gaussian blur, implemented in WebGL fragment shaders for cleaning up the resulting light maps. 

[Demo](https://reubenjcarter.github.io/web3d-lightmapper/)

Obligatory Cornell Box
<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/ReubenJCarter/web3d-lightmapper/master/pictures/corenllbox.jpg"/>
</p>

Lightmap of Cornell Box
<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/ReubenJCarter/web3d-lightmapper/master/pictures/cornellbox-lightmap.jpg"/>
</p>

This is currently a work in progress, there are many issues and much I still need to add.

The project makes use of the following libraries; 
- xatlas library created by Jonathan Young
- bvh library created by Arsène Pérard-Gayot
- glm for linear algebra created by G-Truc Creation
- xoshiro128++ for fast random numbers by David Blackman and Sebastiano Vigna

