/**
 * 
 * Takes a set of meshes from JS world.
 * For each mesh, generate optimal lighting UV coordinates. 
 * Send the UVs back to JS world
 * 
 */

#include <iostream>
#include <string>

#include <emscripten/emscripten.h>
#include <emscripten/bind.h>

#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4

#include "uvUnwrapper.hpp"
#include "renderer.hpp"
#include "bvh.hpp"

using namespace emscripten;

void TestFunction()
{
	std::cout << "HELLO WORLD FROM WASM!!\n"; 
}


class LightMapper
{
public:
    BVH bvh;
    Renderer renderer; 
    UVUnwrapper uVUnwrapper; 

    LightMapper()
    {
        
    }

    bool Init(std::string canvasId)
    {
        return renderer.Init(canvasId); 
    }

    void Start(std::vector<float>& positions, 
                         std::vector<float>& normals, 
                         std::vector<float>& uvs,
                         std::vector<float>& lmuvs,
                         std::vector<unsigned int>& indices,
                         std::vector<unsigned int>& meshes, 
                         std::vector<float>& transforms, 
                         std::vector<std::string>& materials, 
                         int width, 
                         int height, 
                         int frameCount, 
                         int maxBounces)
    {
        std::cout << "LightMapper Init\n"; 
        std::cout << positions.size()/3 << " vericies, " << indices.size()/3 << " triangles, " << meshes.size()/4 << " meshes" << "\n";

        //
        //Transform all verticies using transformation matrix (it is in vec4, vec4 vec4, vec4   order like glsl), also calc vcount for each mesh
        for(int m = 0; m < meshes.size() / 4; m++)
        {
            uint32_t voffset = meshes[m*4 + 2];
            uint32_t vcount = meshes[m*4 + 3];

            glm::mat4 t = glm::mat4(transforms[m * 16 + 0], transforms[m * 16 + 1], transforms[m * 16 + 2], transforms[m * 16 + 3],
                                    transforms[m * 16 + 4], transforms[m * 16 + 5], transforms[m * 16 + 6], transforms[m * 16 + 7],
                                    transforms[m * 16 + 8], transforms[m * 16 + 9], transforms[m * 16 + 10], transforms[m * 16 + 11],
                                    transforms[m * 16 + 12], transforms[m * 16 + 13], transforms[m * 16 + 14], transforms[m * 16 + 15] );
            glm::mat4 tn = glm::transpose(glm::inverse(t));
            
            for(uint32_t i = voffset; i < voffset + vcount; i++)
            {
                uint32_t inx = i; 
                glm::vec4 pos = t * glm::vec4( positions[inx*3+0], positions[inx*3+1], positions[inx*3+2], 1); 
                glm::vec3 norm = tn * glm::vec4( normals[inx*3+0], normals[inx*3+1], normals[inx*3+2], 0);  
                norm = glm::normalize(norm); 
                positions[inx*3+0] = pos.x; 
                positions[inx*3+1] = pos.y; 
                positions[inx*3+2] = pos.z;
                normals[inx*3+0] = norm.x; 
                normals[inx*3+1] = norm.y; 
                normals[inx*3+2] = norm.z; 
            }
        }
        
        //
        //Lightmap pack the whole scene 
        uVUnwrapper.Run(positions, normals, uvs, lmuvs, indices, meshes); 

        //
        //construct BVH
        bvh.Construct(positions, indices); 

        //
        //Start renderer
        renderer.Start(positions, normals, lmuvs, indices, meshes, materials, bvh, width, height, frameCount, maxBounces);
    }

    void SetMainLight(
        float dirX, float dirY, float dirZ,
        float colR, float colG, float colB, 
        float power, float radius
    )
    {
        
        renderer.SetMainLight( glm::vec3(dirX, dirY, dirZ), glm::vec3(colR, colG, colB), power, radius ); 
    }

    void SetAmbientLight(
        float skyColR, float skyColG, float skyColB, 
        float skyIntensity, 
        float groundColR, float groundColG, float groundColB, 
        float groundIntensity
    )
    {
        renderer.SetAmbientLight(glm::vec3(skyColR, skyColG, skyColB), skyIntensity, glm::vec3(groundColR, groundColG, groundColB), groundIntensity); 
    }

    void Stop() 
    {
        renderer.Stop(); 
    }

    void RenderLightMapFrames(int framesToRender)
    {
        renderer.RenderLightMapFrames(framesToRender); 
    }

    void AverageSamples()
    {
        renderer.AverageSamples(); 
    }

    void Dilate()
    {
        renderer.Dilate();
    }

    void RunGaussFilter()
    {
        renderer.RunGaussFilter(); 
    }

    void RunEdgeAwareFilter()
    {
        renderer.RunEdgeAwareFilter();
    }

    void RunMedianFilter()
    {
        renderer.RunMedianFilter(); 
    }

    void RenderToCanvas() 
    {
        renderer.RenderToCanvas(); 
    }

    void AverageSamplesAndRenderToCanvas()
    {
        renderer.AverageSamplesAndRenderToCanvas(); 
    }

}; 


EMSCRIPTEN_BINDINGS(LightMapperWASM) 
{
    function("TestFunction", &TestFunction);

    class_<LightMapper>("LightMapper")
    .constructor<>()
    .function("Init", &LightMapper::Init)
    .function("Start", &LightMapper::Start)
    .function("SetMainLight", &LightMapper::SetMainLight)
    .function("SetAmbientLight", &LightMapper::SetAmbientLight)
    .function("Stop", &LightMapper::Stop)
    .function("RenderLightMapFrames", &LightMapper::RenderLightMapFrames)
    .function("AverageSamples", &LightMapper::AverageSamples)
    .function("Dilate", &LightMapper::Dilate)
    .function("RunGaussFilter", &LightMapper::RunGaussFilter)
    .function("RunEdgeAwareFilter", &LightMapper::RunEdgeAwareFilter)
    .function("RunMedianFilter", &LightMapper::RunMedianFilter)
    .function("RenderToCanvas", &LightMapper::RenderToCanvas)
    .function("AverageSamplesAndRenderToCanvas", &LightMapper::AverageSamplesAndRenderToCanvas); 

    register_vector<int>("VectorInt");
    register_vector<unsigned int>("VectorUInt");
    register_vector<float>("VectorFloat");
    register_vector<std::string>("VectorString");
}