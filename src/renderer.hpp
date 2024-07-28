/**
 * 
 * Renderer 
 * 
 * Could this be useful? https://alain.xyz/blog/machine-learning-denoising#ref_schied2019
 * 
 * Blue noise could be very important 
 * https://blog.tuxedolabs.com/2018/12/07/the-importance-of-good-noise.html
 * https://www.gdcvault.com/play/1026185/
 * 
 * Cosine Sampling
 * https://www.rorydriscoll.com/2009/01/07/better-sampling/
 * 
 * SVGF filtering 
 * 
 */

#pragma once

#include "bvh.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <random>
#include <chrono>


#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#include <GLES3/gl3.h>
//#include <GLES2/gl2.h>


#include "fastRand.hpp"
#include "misc.hpp"

#define PI 3.1415926538

class Renderer
{
private:

    //
    //
    
    std::string vertPostProcSrc = R"(#version 300 es

        in vec3 position; 
        out vec2 vPosition;

        void main()
        {
            vPosition = vec2(position.x, position.y); 
            vec2 finalP = 2.0 * vec2(position.x - 0.5, position.y - 0.5); 
            gl_Position = vec4(finalP.x, finalP.y, 0.0, 1.0); 
        }
    )"; 

    std::string fragPostProcPassSrc = R"(#version 300 es

        precision highp float;
        precision highp int;

        in vec2 vPosition;
        out vec4 outColor;

        uniform int imgWidth;
        uniform int imgHeight; 

        uniform sampler2D img; 

        void main()
        {
            ivec2 coord = ivec2( int(vPosition.x * float(imgWidth)), int(vPosition.y * float(imgHeight)) );
            vec4 texel = texelFetch(img, coord, 0); 
            outColor = vec4(texel.x, texel.y, texel.z, 1.0); 
        }
    )"; 

    std::string fragPostProcAccumulateSrc = R"(#version 300 es
        
        #define PI 3.1415926538

        precision highp float;
        precision highp int;

        in vec2 vPosition;
        out vec4 outColor;

        uniform int imgWidth;
        uniform int imgHeight; 
        uniform int sampleCount; 

        uniform sampler2D img; 

        //Average and multiply by PI https://cseweb.ucsd.edu/classes/sp17/cse168-a/CSE168_13_PathTracing2.pdf
        void main()
        {
            ivec2 coord = ivec2( int(vPosition.x * float(imgWidth)), int(vPosition.y * float(imgHeight)) );

            vec4 texel = texelFetch(img, coord, 0); 
            
            vec3 finalColor = (PI / float(sampleCount)) * texel.xyz; 

            outColor = vec4(finalColor.x, finalColor.y, finalColor.z, texel.w); 
        }
    )";

    std::string fragPostProcDilateSrc = R"(#version 300 es

        #define EPS 0.5
        
        precision highp float;
        precision highp int;

        in vec2 vPosition;
        out vec4 outColor;

        uniform int imgWidth;
        uniform int imgHeight; 

        uniform sampler2D img; 

        void main()
        {
            ivec2 coord = ivec2( int(vPosition.x * float(imgWidth)), int(vPosition.y * float(imgHeight)) );
            
            vec4 texel11 = texelFetch(img, coord, 0);
            vec4 outputTexel;
            if(texel11.a < EPS) {
                //vec4 texel00 = texelFetch(img, coord - ivec2(-1, -1), 0); 
                vec4 texel01 = texelFetch(img, coord - ivec2( 0, -1), 0); 
                //vec4 texel02 = texelFetch(img, coord - ivec2( 1, -1), 0); 

                vec4 texel10 = texelFetch(img, coord - ivec2(-1, 0), 0);                 
                vec4 texel12 = texelFetch(img, coord - ivec2( 1, 0), 0); 

                //vec4 texel20 = texelFetch(img, coord - ivec2(-1, 1), 0); 
                vec4 texel21 = texelFetch(img, coord - ivec2( 0, 1), 0); 
                //vec4 texel22 = texelFetch(img, coord - ivec2( 1, 1), 0); 

                float alphaSum = texel01.w + texel10.w + texel12.w + texel21.w;
                if(alphaSum > EPS){
                    outputTexel = vec4( (texel01.xyz + texel10.xyz + texel12.xyz + texel21.xyz) * (1.0 / alphaSum), 1.0); 
                }
                else {
                    outputTexel = vec4(0.0, 0.0, 0.0, 0.0); 
                }
                
            }
            else {
                outputTexel = texel11; 
            }

            outColor = outputTexel; 
        }
    )"; 

    std::string fragPostProcGaussSrc = R"(#version 300 es

        precision highp float;
        precision highp int;

        in vec2 vPosition;
        out vec4 outColor;

        uniform int imgWidth;
        uniform int imgHeight; 

        uniform sampler2D img; 

        void main()
        {
            ivec2 coord = ivec2( int(vPosition.x * float(imgWidth)), int(vPosition.y * float(imgHeight)) );
            
            vec4 texel00 = texelFetch(img, coord - ivec2(-1, -1), 0); 
            vec4 texel01 = texelFetch(img, coord - ivec2( 0, -1), 0); 
            vec4 texel02 = texelFetch(img, coord - ivec2( 1, -1), 0); 
            
            vec4 texel10 = texelFetch(img, coord - ivec2(-1, 0), 0); 
            vec4 texel11 = texelFetch(img, coord - ivec2( 0, 0), 0); 
            vec4 texel12 = texelFetch(img, coord - ivec2( 1, 0), 0); 

            vec4 texel20 = texelFetch(img, coord - ivec2(-1, 1), 0); 
            vec4 texel21 = texelFetch(img, coord - ivec2( 0, 1), 0); 
            vec4 texel22 = texelFetch(img, coord - ivec2( 1, 1), 0); 
            
            vec4 outputTexel = 
                texel00 * 0.0625 + texel01 * 0.125 + texel02 * 0.0625 + 
                texel10 * 0.125  + texel11 * 0.25  + texel12 * 0.125  + 
                texel20 * 0.0625 + texel21 * 0.125 + texel22 * 0.0625; 
            
            outColor = vec4(outputTexel.x, outputTexel.y, outputTexel.z, 1.0); 
        }
    )"; 

    std::string fragPostProcMedianSrc = R"(#version 300 es

        precision highp float;
        precision highp int;

        //Sorting network
        #define vec vec3
        #define toVec(x) x.rgb

        #define s2(a, b)				temp = a; a = min(a, b); b = max(temp, b);
        #define mn3(a, b, c)			s2(a, b); s2(a, c);
        #define mx3(a, b, c)			s2(b, c); s2(a, c);

        #define mnmx3(a, b, c)			mx3(a, b, c); s2(a, b);                                   // 3 exchanges
        #define mnmx4(a, b, c, d)		s2(a, b); s2(c, d); s2(a, c); s2(b, d);                   // 4 exchanges
        #define mnmx5(a, b, c, d, e)	s2(a, b); s2(c, d); mn3(a, c, e); mx3(b, d, e);           // 6 exchanges
        #define mnmx6(a, b, c, d, e, f) s2(a, d); s2(b, e); s2(c, f); mn3(a, b, c); mx3(d, e, f); // 7 exchanges

        in vec2 vPosition;
        out vec4 outColor;

        uniform int imgWidth;
        uniform int imgHeight; 

        uniform sampler2D img; 

        void main()
        {
            ivec2 coord = ivec2( int(vPosition.x * float(imgWidth)), int(vPosition.y * float(imgHeight)) );

            vec v[6];

            v[0] = toVec( texelFetch(img, coord + ivec2(-1, -1), 0 ) );
            v[1] = toVec( texelFetch(img, coord + ivec2( 0, -1), 0 ) );
            v[2] = toVec( texelFetch(img, coord + ivec2(+1, -1), 0 ) );
            v[3] = toVec( texelFetch(img, coord + ivec2(-1,  0), 0 ) );
            v[4] = toVec( texelFetch(img, coord + ivec2( 0,  0), 0 ) );
            v[5] = toVec( texelFetch(img, coord + ivec2(+1,  0), 0 ) );

            // Starting with a subset of size 6, remove the min and max each time
            vec temp;
            mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);

            v[5] = toVec( texelFetch(img, coord + ivec2(-1, +1), 0 ) );

            mnmx5(v[1], v[2], v[3], v[4], v[5]);

            v[5] = toVec( texelFetch(img, coord + ivec2( 0, +1), 0 ) );

            mnmx4(v[2], v[3], v[4], v[5]);

            v[5] = toVec( texelFetch(img, coord + ivec2(+1, +1), 0 ) );

            mnmx3(v[3], v[4], v[5]);
            //toVec(gl_FragColor) = v[4];

            outColor = vec4(v[4].x, v[4].y, v[4].z, 1.0); 
        }
    )"; 

    std::string fragEdgeAwareSrc = R"(#version 300 es

        precision highp float;
        precision highp int;

        in vec2 vPosition;
        out vec4 outColor;

        uniform int imgWidth;
        uniform int imgHeight; 

        uniform sampler2D img; 
        uniform sampler2D normMap;

        void main()
        {
            ivec2 coord = ivec2( int(vPosition.x * float(imgWidth)), int(vPosition.y * float(imgHeight)) );
            

            vec4 img00 = texelFetch(img, coord - ivec2(-1, -1), 0); 
            vec4 img01 = texelFetch(img, coord - ivec2( 0, -1), 0); 
            vec4 img02 = texelFetch(img, coord - ivec2( 1, -1), 0); 
            
            vec4 img10 = texelFetch(img, coord - ivec2(-1, 0), 0); 
            vec4 img11 = texelFetch(img, coord - ivec2( 0, 0), 0); 
            vec4 img12 = texelFetch(img, coord - ivec2( 1, 0), 0); 

            vec4 img20 = texelFetch(img, coord - ivec2(-1, 1), 0); 
            vec4 img21 = texelFetch(img, coord - ivec2( 0, 1), 0); 
            vec4 img22 = texelFetch(img, coord - ivec2( 1, 1), 0); 


            vec4 norm00 = texelFetch(normMap, coord - ivec2(-1, -1), 0); 
            vec4 norm01 = texelFetch(normMap, coord - ivec2( 0, -1), 0); 
            vec4 norm02 = texelFetch(normMap, coord - ivec2( 1, -1), 0); 
            
            vec4 norm10 = texelFetch(normMap, coord - ivec2(-1, 0), 0); 
            vec4 norm11 = texelFetch(normMap, coord - ivec2( 0, 0), 0); 
            vec4 norm12 = texelFetch(normMap, coord - ivec2( 1, 0), 0); 

            vec4 norm20 = texelFetch(normMap, coord - ivec2(-1, 1), 0); 
            vec4 norm21 = texelFetch(normMap, coord - ivec2( 0, 1), 0); 
            vec4 norm22 = texelFetch(normMap, coord - ivec2( 1, 1), 0); 
            
            

            vec4 outputTexel = 
                img00 * 0.0625 + img01 * 0.125 + img02 * 0.0625 + 
                img10 * 0.125  + img11 * 0.25  + img12 * 0.125  + 
                img20 * 0.0625 + img21 * 0.125 + img22 * 0.0625; 
            

            outColor = vec4(outputTexel.x, outputTexel.y, outputTexel.z, 1.0); 
            

            //outColor = vec4(texelFetch(normMap, coord, 0).xyz, 1.0); 
            //outColor = vec4(texelFetch(img, coord, 0).xyz, 1.0); 
        }
    )"; 

    //
    //PATH TRACING SHADERS

    std::string vertPathTraceSrc = R"(#version 300 es

        in vec3 position; 
        in vec3 normal; 
        in vec2 lmuv;

        out vec3 vPosition;
        out vec3 vNormal;
        out vec2 vLmuv; 

        void main()
        {
            vPosition = position; 
            vNormal = normal; 
            vLmuv = lmuv; 

            vec2 finalP = 2.0 * vec2(lmuv.x - 0.5, lmuv.y - 0.5); 
            gl_Position = vec4(finalP.x, finalP.y, 0.0, 1.0); 
        }
    )"; 

    std::string fragNormalMapSrc = R"(#version 300 es
        precision highp float;
        precision highp int;

        in vec3 vPosition;
        in vec3 vNormal;
        in vec2 vLmuv;
        out vec4 outColor;

        void main()
        {
            vec3 norm = 0.5 * normalize(vNormal) + vec3(0.5); 
            outColor = vec4(norm.x, norm.y, norm.z, 1.0); 
        }
    )";

    //https://cseweb.ucsd.edu/classes/sp17/cse168-a/CSE168_13_PathTracing2.pdf
    std::string fragPathTraceSrc = R"(#version 300 es
        #define PI 3.1415926538
        #define GOLDEN_RATIO 1.618033988749

        precision highp float;
        precision highp int;
        precision highp usampler2D;
        precision highp sampler2DArray;

        in vec3 vPosition;
        in vec3 vNormal;
        in vec2 vLmuv;

        out vec4 outColor;


        uniform sampler2DArray noiseTex; 
        uniform sampler2D positionsTex; 
        uniform sampler2D normalsTex; 
        uniform usampler2D trianglesTex; 
        uniform sampler2D materialColorsTex; 

        uniform sampler2D bvhBoundsTex; 
        uniform usampler2D bvhNodeDataTex; 
        
        uniform sampler2D accumulationTex; 

        uniform int texBufferWidth; 
        uniform int maxBounces;  
        uniform int triangleCount; 
        uniform int bvhNodeCount; 
        uniform int currentFrame; 
        uniform int frameBufferWidth; 
        uniform int frameBufferHeight; 
        uniform int frameCount; 
        uniform int randomSamplingMode; 

        uniform vec3 mainLightDirection; 
        uniform vec3 mainLightColor; 
        uniform float mainLightIntensity; 
        uniform float mainLightRadius; 

        uniform vec3 ambientLightSkyColor; 
        uniform float ambientLightSkyIntensity; 
        uniform vec3 ambientLightGroundColor; 
        uniform float ambientLightGroundIntensity; 


        vec4 TexelFFromInx(sampler2D tex, int inx)
        {
            int y = inx / texBufferWidth;
            int x = inx % texBufferWidth; 
            return texelFetch(tex, ivec2(x, y), 0); 
        }

        uvec4 TexelUFromInx(usampler2D tex, int inx)
        {
            int y = inx / texBufferWidth; 
            int x = inx % texBufferWidth; 
            return texelFetch(tex, ivec2(x, y), 0); 
        }





        vec2 UniformRandomRadiusAngle(vec2 f, int currentSample, int currentBounce)
        {
            ivec2 noiseTexSize = textureSize(noiseTex, 0).xy; 
            vec2 noiseTexScale = vec2(float(frameBufferWidth) / float(noiseTexSize.x), float(frameBufferHeight) / float(noiseTexSize.y)); 
            vec4 noisev = texture(noiseTex, vec3(f * noiseTexScale, float(currentBounce)) ); 
            return noisev.xy;
        }

        vec2 FixedSampleRadiusAngle(vec2 f, int currentSample, int currentBounce)
        {
            ivec2 noiseTexSize = textureSize(noiseTex, 0).xy; 
            vec2 noiseTexScale = vec2(float(frameBufferWidth) / float(noiseTexSize.x), float(frameBufferHeight) / float(noiseTexSize.y)); 
            vec4 noisev = texture(noiseTex, vec3(f * noiseTexScale, float(currentBounce)) ); 
            
            int count = int( sqrt( float(frameCount) )); 
            int k = (currentSample);
            float xx = float(k%count) / float(count); 
            float yy = float(k/count) / float(count); 

            return 0.9*vec2( xx, yy ) + 0.1 * noisev.xy; 
        }

        vec2 GetRadiusAngleSample(vec2 f, int currentSample, int currentBounce)
        {
            if(randomSamplingMode == 1)
                return FixedSampleRadiusAngle(f, currentSample, currentBounce); 
            else
                return UniformRandomRadiusAngle(f, currentSample, currentBounce); 
        }



        //https://www.rorydriscoll.com/2009/01/07/better-sampling/
        vec3 CosineHemi(vec2 randomVal, vec3 norm)
        {
            float r = sqrt(randomVal.x);
            float theta = 2.0 * PI * randomVal.y;
        
            float x = r * cos(theta);
            float y = r * sin(theta);
        
            vec3 rh = vec3(x, y, sqrt(max(0.0, 1.0 - randomVal.x)) );

            //compute norm and tangent of samples 
            vec3 n = vec3(0.0, 0.0, 1.0); 
            vec3 u = cross(n, norm);
            
            //area of parallelagram is zero 
            float sina = length(u); 
            if(sina < 0.00001)
                u = vec3(1.0, 0.0, 0.0); 
            else 
                u = normalize(u);

            //u is the axis or rotation, a is cos of the angle or rotation
            //construct rotation matrix from wiki....https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
            float cosa = dot(n, norm);
            float onemincosa = 1.0 - cosa; 
            float uyux = u.y * u.x; 
            float uzux = u.z * u.x;
            float uyuz = u.y * u.z;

            vec3 m0 = vec3(cosa + u.x * u.x * onemincosa, 
                           uyux * onemincosa + u.z * sina, 
                           uzux * onemincosa - u.y * sina); 

            vec3 m1 = vec3(uyux * onemincosa - u.z * sina, 
                           cosa + u.y * u.y * onemincosa, 
                           uyuz * onemincosa + u.x * sina); 

            vec3 m2 = vec3(uzux * onemincosa + u.y * sina,
                           uyuz * onemincosa - u.x * sina,
                           cosa + u.z * u.z * onemincosa); 

            mat3 R = mat3(m0, m1, m2); 

            return  R * rh; 
        }



        vec3 SkyBackgroundEmittance(vec3 rayDir) 
        {
            //Ambient Light
            vec3 ambientLightGround = ambientLightGroundColor * ambientLightGroundIntensity; 
	        vec3 ambientLightSky = ambientLightSkyColor * ambientLightSkyIntensity; 

	        float ambientLightA = 0.5 + 0.5 * dot(rayDir, vec3(0, 1, 0) );
            vec3 ambientLight = mix(ambientLightGround, ambientLightSky, ambientLightA); 


            //main directional Light
            vec3 mainLight = mainLightIntensity * mainLightColor; 
            vec3 mainLightVec = -normalize(mainLightDirection);
            float mainLightFactor = dot(rayDir, mainLightVec) > cos(mainLightRadius) ? 1.0: 0.0; 
            
            //Mix light together. 
            return mix(ambientLight, mainLight, mainLightFactor); 
        }



        bool IntersectAABB(vec3 rayOrig, vec3 rayDirInv, vec3 minaabb, vec3 maxaabb, out float hitDist)
        {
            vec3 t1 = (minaabb - rayOrig) * rayDirInv;
            vec3 t2 = (maxaabb - rayOrig) * rayDirInv;
            float tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
            float tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
            hitDist = tmin;
            return tmax >= max(0.0f, tmin);
        }

        bool IntersectTriangle(
            vec3 rayOrig, vec3 rayDir, 
            vec3 va, vec3 vb, vec3 vc,
            out vec4 hit)
        {
            vec3 e0 = vb - va;
            vec3 e1 = vc - va;

            vec4 hitTemp;
            vec3 pv = cross(rayDir, e1);
            float det = dot(e0, pv);
            vec3 tv = rayOrig - va;
            vec3 qv = cross(tv, e0);
            hitTemp.xyz = vec3(dot(e1, qv), dot(tv, pv), dot(rayDir, qv)) / det;
            hitTemp.w = hitTemp.x;
            hitTemp.x = 1.0f - hitTemp.z - hitTemp.y;
            bool triray = all(greaterThanEqual(hitTemp, vec4(0.0f, 0.0f, 0.0f, 0.0001f)));
            hit = hitTemp;

            return triray; 
        }


        int stack[64];
        void RayIntersectBVHStack(vec3 rayOrig, vec3 rayDir, out int triIndex, out vec4 hit)
        {
            vec3 rayDirInv = vec3(1.0f, 1.0f, 1.0f) / rayDir;
            int nodeIndex = 0;

            float dist = 1e10;
            triIndex = -1;

            //set up traversal stack 
            int stackPtr = 0;
        	stack[0] = 0;

            while(stackPtr >= 0)
            {
                //pop node from stack
        		nodeIndex = stack[stackPtr--];

                //Get node data
                uvec2 n = TexelUFromInx(bvhNodeDataTex, nodeIndex).xy; 
                uint first_child_or_primitive = n.x; 
                uint primitive_count = n.y; 
                bool isLeaf = primitive_count != uint(0); 

                if(!isLeaf)
                {
                    //Get node aabb
                    vec3 minaabb = TexelFFromInx(bvhBoundsTex, 2 * nodeIndex).xyz; 
                    vec3 maxaabb = TexelFFromInx(bvhBoundsTex, 2 * nodeIndex + 1).xyz; 
    
                    //AABB Ray intersection 
                    vec3 t1 = (minaabb - rayOrig) * rayDirInv;
                    vec3 t2 = (maxaabb - rayOrig) * rayDirInv;
                    float tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
                    float tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
                    bool raybox = tmax >= max(0.0f, tmin) && tmin < dist;
                    if(raybox) //if the ray hits the BVH box
                    {
                        stack[++stackPtr] = int(first_child_or_primitive); //add right child to stack
				        stack[++stackPtr] = int(first_child_or_primitive) + 1; //add left child to stack
                    }
                }
                else
                {
                    for(uint i = first_child_or_primitive; i < first_child_or_primitive + primitive_count; i++)
                    {
                        //Get the triangle data
                        uint triangleIndex = i; 
                        uvec4 tri = TexelUFromInx(trianglesTex, int(triangleIndex) );
                        vec3 va = TexelFFromInx(positionsTex, int(tri.x)).xyz;
                        vec3 vb = TexelFFromInx(positionsTex, int(tri.y)).xyz;
                        vec3 vc = TexelFFromInx(positionsTex, int(tri.z)).xyz;

                        //Triangle Ray Intersection
                        vec4 hitTemp;
                        bool triray = IntersectTriangle(rayOrig, rayDir, va, vb, vc, hitTemp);
                        
                        if(triray  && hitTemp.w < dist)
                        {
                            dist = hitTemp.w;
                            triIndex = int(i);
                            hit = hitTemp;
                        }
                    }
                }
            }
        }


        void RayIntersectBVHStackless(vec3 rayOrig, vec3 rayDir, out int triIndex, out vec4 hit)
        {
            vec3 rayDirInv = vec3(1.0f, 1.0f, 1.0f) / rayDir;
            int nodeIndex = 0;

            float dist = 1e10;
            triIndex = -1;

            while(nodeIndex < bvhNodeCount)
            {
                //Get node data
                uvec3 n = TexelUFromInx(bvhNodeDataTex, nodeIndex).xyz; 
                uint first_child_or_primitive = n.x; 
                uint primitive_count = n.y; 
                uint branchSkipInx = n.z; 
                bool isLeaf = primitive_count != uint(0); 

                //Get node aabb
                vec3 minaabb = TexelFFromInx(bvhBoundsTex, 2 * nodeIndex).xyz; 
                vec3 maxaabb = TexelFFromInx(bvhBoundsTex, 2 * nodeIndex + 1).xyz; 

                //AABB Ray intersection 
                vec3 t1 = (minaabb - rayOrig) * rayDirInv;
                vec3 t2 = (maxaabb - rayOrig) * rayDirInv;
                float tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
                float tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
                bool raybox = tmax >= max(0.0f, tmin) && tmin < dist;
                
                //Which node to move onto next. if the current one is a leaf, one of its siblings...
                //if the ray hits the node and its not a leaf, then need to go through all its children, so go to first sibling
                //if not just skip the whole branch of the tree. 
                nodeIndex = !isLeaf && raybox ? int(first_child_or_primitive) : int(branchSkipInx);

                if(raybox)
                {
                    if(isLeaf)
                    {
                        for(uint i = first_child_or_primitive; i < first_child_or_primitive + primitive_count; i++)
                        {
                            //Get the triangle data
                            uint triangleIndex = i; 
                            uvec4 tri = TexelUFromInx(trianglesTex, int(triangleIndex) );
                            vec3 va = TexelFFromInx(positionsTex, int(tri.x)).xyz;
                            vec3 vb = TexelFFromInx(positionsTex, int(tri.y)).xyz;
                            vec3 vc = TexelFFromInx(positionsTex, int(tri.z)).xyz;

                            //Triangle Ray Intersection
                            vec4 hitTemp;
                            bool triray = IntersectTriangle(rayOrig, rayDir, va, vb, vc, hitTemp);

                            if(triray && hitTemp.w < dist)
                            {
                                dist = hitTemp.w;
                                triIndex = int(i);
                                hit = hitTemp;
                            }
                        }
                    }
                }


            }
        }


        void RayIntersectBrute(vec3 rayOrig, vec3 rayDir, out int triIndex, out vec4 hit)
        {
            float dist = 1e10;
            triIndex = -1;
            vec4 hitTemp;

            for(int i = 0; i < triangleCount; i++)
            {
                uvec4 tri = TexelUFromInx(trianglesTex, i);
                vec3 va = TexelFFromInx(positionsTex, int(tri.x)).xyz;
                vec3 vb = TexelFFromInx(positionsTex, int(tri.y)).xyz;
                vec3 vc = TexelFFromInx(positionsTex, int(tri.z)).xyz;

                //Triangle Ray Intersection
                vec4 hitTemp;
                bool triray = IntersectTriangle(rayOrig, rayDir, va, vb, vc, hitTemp);
                
                if(triray  && hitTemp.w < dist)
                {
                    dist = hitTemp.w;
                    triIndex = i;
                    hit = hitTemp;
                }
            }
        }



        void main()
        {
            
            vec3 vNorm = normalize(vNormal);

            
            vec3 rayOrig = vPosition; 
            vec3 rayDir = CosineHemi( GetRadiusAngleSample(vLmuv, currentFrame, 0) , vNorm); 
            
            vec3 runningReflectanceFactor = vec3(1.0, 1.0, 1.0);
            vec3 sampleBrightness = vec3(0.0, 0.0, 0.0); 
   
            for(int b = 0; b < maxBounces; b++)
            {

                int triInx = -1; 
                vec4 hit; 
                RayIntersectBVHStackless(rayOrig, rayDir, triInx, hit);

                if(triInx < 0)
                {
                    sampleBrightness += runningReflectanceFactor * SkyBackgroundEmittance(rayDir); 
                    break; 
                }
                else
                {
                    uvec4 tri = TexelUFromInx(trianglesTex, triInx);

                    vec3 norma = (TexelFFromInx(normalsTex, int(tri.x)).xyz);	
                    vec3 normb = (TexelFFromInx(normalsTex, int(tri.y)).xyz);
                    vec3 normc = (TexelFFromInx(normalsTex, int(tri.z)).xyz);
                    vec3 norm = normalize(vec3(norma * hit.x + normb * hit.y + normc * hit.z));

                    vec3 posa = TexelFFromInx(positionsTex, int(tri.x)).xyz;
                    vec3 posb = TexelFFromInx(positionsTex, int(tri.y)).xyz;
                    vec3 posc = TexelFFromInx(positionsTex, int(tri.z)).xyz;
                    vec3 pos = vec3(posa * hit.x + posb * hit.y + posc * hit.z);

                    vec3 newRayO = vec3(0, 0, 0);
                    vec3 newRayD = vec3(0, 0, 0);

                    vec3 materialEmittance = vec3(0, 0, 0); 
                    sampleBrightness += runningReflectanceFactor * materialEmittance;

                    if(b < maxBounces-1) //dont bother with random direction if its the last bounce
                    {
                        newRayO = pos;
                        newRayD = CosineHemi( UniformRandomRadiusAngle(vLmuv, currentFrame, b+1) , norm);
                    
                        vec3 albedoColor = TexelFFromInx(materialColorsTex, int(tri.w)).xyz;
                        vec3 reflectanceFactor = albedoColor;
                        runningReflectanceFactor *= reflectanceFactor;
                    }                    

                    rayOrig = newRayO;
                    rayDir = newRayD;
                }
            }
            
            vec4 acc; 
            if(currentFrame == 0)
            {
                acc = vec4(0, 0, 0, 1); 
            }
            else 
            {
                ivec2 coord = ivec2( int(vLmuv.x * float(frameBufferWidth)), int(vLmuv.y * float(frameBufferHeight)) );
                acc = texelFetch(accumulationTex, coord, 0);
            }

            outColor = vec4(sampleBrightness.xyz + acc.xyz, 1); 

            //vec2 rndn = UniformRandomRadiusAngle(vLmuv, 0, 0); 
            //outColor = vec4(rndn.x, rndn.y, 1.0, 1.0); 
        }
    )"; 

    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context;
    GLuint renderProgram; 
    GLuint accumulateProgram; 
    GLuint normalMapProgram; 
    GLuint postProcMedianProgram; 
    GLuint postProcPassThroughProgram; 
    GLuint postProcGaussProgram; 
    GLuint postProcDilateProgram; 
    GLuint postProcEdgeAwareProgram;

    GLuint noiseTex = 0; 

    //FrameBuffer
    GLuint frameBufferTexture = 0;
    GLuint frameBuffer = 0;  
    GLuint frameBuffer2Texture = 0;
    GLuint frameBuffer2 = 0;  

    GLuint normalFrameBufferTexture = 0; 
    GLuint frameBufferNormal = 0;  
    GLuint positionFrameBufferTexture = 0;
    GLuint frameBufferPosition = 0;   

    //Mesh Data textures
    GLuint positionsTex = 0; 
    GLuint normalsTex = 0; 
    GLuint trianglesTex = 0; 
    GLuint materialColorsTex = 0; 

    //BVH Data Textures
    GLuint bvhBoundsTex = 0; 
    GLuint bvhNodeDataTex = 0; 

    //
    const int texBufferWidth = 1024; 
    const int noiseTexSize = 256;

    int maxBounces = 1;

    int frameBufferTextureWidth = 0; 
    int frameBufferTextureHeight = 0;

    int renderCount = 0; 
    int currentFrame = 0; 
    bool normalMapHasRun = false;

    unsigned int indexCount;
    int triangleCount; 
    int bvhNodeCount; 
    uint32_t positionSize;
    uint32_t normalSize;
    uint32_t lmuvSize;
    int frameCount; 

    //VBO
    GLuint vbo = 0;
    GLuint ebo = 0; 
    GLuint vboPostProc = 0; 
    GLuint eboPostProc = 0; 

    //random 
    std::minstd_rand randGenerator; 
    std::uniform_real_distribution<float> randDistributionF;
    FastRand fastRand; 

    //Main Light
    glm::vec3 mainLightDirection = glm::vec3(0.5f, -1.0f, 0.5f); 
    glm::vec3 mainLightColor = glm::vec3(1.0f, 1.0f, 1.0f); 
    float mainLightIntensity = 10.0f; 
    float mainLightRadius = 0.1; 

    //Ambient Light
    glm::vec3 ambientLightSkyColor = glm::vec3(1.0f, 1.0f, 1.0f); 
    float ambientLightSkyIntensity = 0.4f; 
    glm::vec3 ambientLightGroundColor = glm::vec3(1.0f, 1.0f, 1.0f); 
    float ambientLightGroundIntensity = 0.2f; 

    GLuint BuildShaderProgram(std::string vertSrc, std::string fragSrc)
    {  
        //Compile vertex and fragment shaders  
        std::cout << "Creating Vertex Shader\n";
        const GLchar* vertSrcs[1];
        GLint vertSrcLens[1];
        vertSrcs[0] = vertSrc.c_str();
        vertSrcLens[0] = vertSrc.length();
        GLuint vertS= glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertS, 1, vertSrcs, vertSrcLens);
        glCompileShader(vertS);
        int vertInfoLen = 0;
        glGetShaderiv( vertS, GL_INFO_LOG_LENGTH, &vertInfoLen );
        if (vertInfoLen > 1) 
        {
            char* infoLog = new char[vertInfoLen];
            glGetShaderInfoLog( vertS, vertInfoLen, NULL, infoLog );
            std::cout << infoLog << "\n"; 
            delete[] infoLog; 
        }
        std::cout << "Creating Fragment Shader\n";
        const GLchar* fragSrcs[1];
        GLint fragSrcLens[1];
        fragSrcs[0] = fragSrc.c_str();
        fragSrcLens[0] = fragSrc.length();
        GLuint fragS = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragS, 1, fragSrcs, fragSrcLens);
        glCompileShader(fragS);
        int fragInfoLen = 0;
        glGetShaderiv( fragS, GL_INFO_LOG_LENGTH, &fragInfoLen );
        if (fragInfoLen > 1) 
        {
            char* infoLog = new char[fragInfoLen];
            glGetShaderInfoLog( fragS, fragInfoLen, NULL, infoLog );
            std::cout << infoLog << "\n"; 
            delete[] infoLog; 
        }

        //Build shader program  
        std::cout << "Creating Shader Program\n";
        GLuint programObject = glCreateProgram();
        glAttachShader(programObject, vertS);
        glAttachShader(programObject, fragS);
		
		//Set binding locations (Needds to actually happen before link )
        glBindAttribLocation(programObject, 0, "position");
        glBindAttribLocation(programObject, 1, "normal");
        glBindAttribLocation(programObject, 2, "lmuv");
		
		
        glLinkProgram(programObject);
        glValidateProgram(programObject);
        GLint isLinked = 0;
        glGetProgramiv(programObject, GL_LINK_STATUS, (int*)&isLinked); 
        if(isLinked == GL_FALSE)
        {
            GLint maxLength = 0;
            glGetProgramiv(programObject, GL_INFO_LOG_LENGTH, &maxLength);
            char* infoLog = new char[fragInfoLen];
            glGetProgramInfoLog(programObject, maxLength, &maxLength, infoLog);
            
            glDeleteProgram(programObject);
            glDeleteShader(vertS);
            glDeleteShader(fragS);

            std::cout << infoLog << std::endl;
            delete[] infoLog; 

            return 0; 
        }

        

        return programObject; 
    }

    

    

    GLuint PackTexture3(std::vector<unsigned char>& data)
    {
        
        //deep copy data 
        int elementCount = data.size() / 3; 
        int width = texBufferWidth; 
        int height = elementCount / width + (elementCount % width > 0 ? 1 : 0); 
        unsigned char* copydata = new unsigned char[width * height * 3]; 
        for(int i = 0; i < data.size(); i++)
            copydata[i] = data[i]; 

        GLuint tex; 

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
       
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, copydata);
        
        delete[] copydata; 

        return tex; 
    }

    GLuint PackTexture3f(std::vector<float>& data)
    {
        
        //deep copy data 
        int elementCount = data.size() / 3; 
        int width = texBufferWidth; 
        int height = elementCount / width + (elementCount % width > 0 ? 1 : 0); 
        float* copydata = new float[width * height * 3]; 
        for(int i = 0; i < data.size(); i++)
            copydata[i] = data[i]; 

        GLuint tex; 

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
       
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, copydata);
        
        delete[] copydata; 

        return tex; 
    }

    GLuint PackTexture4u(std::vector<unsigned int>& data)
    {
        GLuint tex; 

        int elementCount = data.size() / 4; 
        int width = texBufferWidth; 
        int height = elementCount / width + (elementCount % width > 0 ? 1 : 0);         
        unsigned int * copydata = new unsigned int[width * height * 4]; 
        for(int i = 0; i < data.size(); i++)
            copydata[i] = data[i]; 

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

         
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, width, height, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, copydata);

        delete[] copydata; 

        return tex;
    }

    GLuint PackTexture3u(std::vector<unsigned int>& data)
    {
        GLuint tex; 

        int elementCount = data.size() / 3; 
        int width = texBufferWidth; 
        int height = elementCount / width + (elementCount % width > 0 ? 1 : 0);         
        unsigned int * copydata = new unsigned int[width * height * 3]; 
        for(int i = 0; i < data.size(); i++)
            copydata[i] = data[i]; 

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

         
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32UI, width, height, 0, GL_RGB_INTEGER, GL_UNSIGNED_INT, copydata);

        delete[] copydata; 

        return tex;
    }

    GLuint PackTexture2u(std::vector<unsigned int>& data)
    {
        GLuint tex; 

        int elementCount = data.size() / 2; 
        int width = texBufferWidth; 
        int height = elementCount / width + (elementCount % width > 0 ? 1 : 0);         
        unsigned int * copydata = new unsigned int[width * height * 2]; 
        for(int i = 0; i < data.size(); i++)
            copydata[i] = data[i]; 

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

         
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32UI, width, height, 0, GL_RG_INTEGER, GL_UNSIGNED_INT, copydata);

        delete[] copydata; 

        return tex;
    }

    GLuint PackTexture1u(std::vector<unsigned int>& data)
    {
        GLuint tex; 

        int elementCount = data.size(); 
        int width = texBufferWidth; 
        int height = elementCount / width + (elementCount % width > 0 ? 1 : 0);         
        unsigned int * copydata = new unsigned int[width * height]; 
        for(int i = 0; i < data.size(); i++)
            copydata[i] = data[i]; 

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

         
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, copydata);

        delete[] copydata; 

        return tex;
    }

    bool CheckValidContext()
    {
        if(context <= 0)
        {
            std::cout << "Lightmapping failed as there is no valid GL context\n"; 
            return false; 
        }

        return true; 
    }

    std::string GetGlError()
    {
        GLint code = glGetError(); 
        if(code == GL_INVALID_ENUM)
            return "GL_INVALID_ENUM"; 

        else if(code == GL_INVALID_VALUE)
            return "GL_INVALID_VALUE"; 

        else if(code == GL_INVALID_OPERATION)
            return "GL_INVALID_OPERATION"; 

        else if(code == GL_INVALID_FRAMEBUFFER_OPERATION)
            return "GL_INVALID_FRAMEBUFFER_OPERATION"; 

        else if(code == GL_OUT_OF_MEMORY)
            return "GL_OUT_OF_MEMORY";

        else 
            return "GL_NO_ERROR";  
    }
   

    void GenerateNoiseTexture()
    {
        int W = noiseTexSize == 0 ? frameBufferTextureWidth : noiseTexSize; 
        int H = noiseTexSize == 0 ? frameBufferTextureHeight : noiseTexSize;

        //Generate random noise textures
        glGenTextures(1, &noiseTex); 
        glBindTexture(GL_TEXTURE_2D_ARRAY, noiseTex);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RG8, W, H, maxBounces, 0, GL_RG, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    }

    void UpdateNoiseTexture(bool useBlueNoise=false) 
    {
        int W = noiseTexSize == 0 ? frameBufferTextureWidth : noiseTexSize; 
        int H = noiseTexSize == 0 ? frameBufferTextureHeight : noiseTexSize;

        glBindTexture(GL_TEXTURE_2D_ARRAY, noiseTex);
        
        std::vector<unsigned char> v; 
        v.reserve(maxBounces * W * H * 2); 

        if(!useBlueNoise)
        {
            
            for(int i = 0; i < maxBounces * W * H / 2; i++)
            {
                //float rnd = randDistributionF(randGenerator); 
                //float rnd = fastRand.nextFloat(); //is this faster?
                //int val = rnd * 255; 

                //each uint32 has 4 bytes, which is two elements of the texture
                int rnd = fastRand.nextUint(); 
                unsigned char* ptr = (unsigned char*)&rnd; 
                unsigned char b0 = *( ptr ); 
                unsigned char b1 = *( ptr + 1 );
                unsigned char b2 = *( ptr + 2 );
                unsigned char b3 = *( ptr + 3 );
                
                v.push_back( b0 );
                v.push_back( b1 );

                v.push_back( b2 );
                v.push_back( b3 );
            }
        }

        else
        {
           //https://hal.archives-ouvertes.fr/hal-02158423/file/blueNoiseTemporal2019_slides.pdf
        }
        
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, W, H, maxBounces, GL_RG, GL_UNSIGNED_BYTE, v.data());
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    }

public:

    Renderer()
    {

    }

    ~Renderer()
    {
        emscripten_webgl_destroy_context(context);
    }

    bool Init(std::string canvasId)
    {
        std::cout << "Initing Renderer\n"; 

        //
        //Creating a context
        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE oldcontext = emscripten_webgl_get_current_context(); 

        EmscriptenWebGLContextAttributes attribs; 
        emscripten_webgl_init_context_attributes(&attribs); 
        attribs.explicitSwapControl = 0;
        attribs.depth = 0;
        attribs.stencil = 0;
        attribs.antialias = 0;
        attribs.majorVersion = 2;
        attribs.minorVersion = 0;
        attribs.enableExtensionsByDefault = true; 

        std::string canv = "#";
        canv += canvasId;

        std::cout << "Creating context for " << canv << "\n"; 

        context = emscripten_webgl_create_context(canv.c_str(), &attribs);
        if(context <= 0){
            //EMSCRIPTEN_RESULT
            EMSCRIPTEN_RESULT res = (EMSCRIPTEN_RESULT)context; 
            std::cout << "There was a problem creating the context " << res << "\n"; 
            return false; 
        }

        emscripten_webgl_make_context_current(context);
        const char* version = (const char*)glGetString(GL_VERSION);
        std::cout << "context created: " << version << "\n"; 

        //
        //Checks for system compat

        //need compatability for render to float texture 
        int EXT_color_buffer_float_supported = emscripten_webgl_enable_extension(context, "EXT_color_buffer_float"); 
        std::cout << "EXT_color_buffer_float_supported=" << (EXT_color_buffer_float_supported == EM_TRUE) << "\n"; 
        if(EXT_color_buffer_float_supported != EM_TRUE)
        {
            return false; 
        }

        //
        //Set up for rendering 
        renderProgram = BuildShaderProgram(vertPathTraceSrc, fragPathTraceSrc); 
        accumulateProgram = BuildShaderProgram(vertPostProcSrc, fragPostProcAccumulateSrc); 
        normalMapProgram = BuildShaderProgram(vertPathTraceSrc, fragNormalMapSrc);
        postProcMedianProgram = BuildShaderProgram(vertPostProcSrc, fragPostProcMedianSrc); 
        postProcPassThroughProgram = BuildShaderProgram(vertPostProcSrc, fragPostProcPassSrc); 
        postProcGaussProgram = BuildShaderProgram(vertPostProcSrc, fragPostProcGaussSrc);
        postProcDilateProgram = BuildShaderProgram(vertPostProcSrc, fragPostProcDilateSrc);
        postProcEdgeAwareProgram = BuildShaderProgram(vertPostProcSrc, fragEdgeAwareSrc);
        

        //
        //Return to old context
        std::cout << "Returning Old Context \n"; 
        emscripten_webgl_make_context_current(oldcontext);

        std::cout << "Done\n"; 
        return true;
    }

    void SetMainLight(glm::vec3 direction, glm::vec3 color, float power, float radius)
    {
        float surfaceArea = radius * radius * PI; 
        float intensity = power / surfaceArea; 

        mainLightDirection = direction; 
        mainLightColor = color; 
        mainLightIntensity = intensity; 
        mainLightRadius = radius; 
    }

    void SetAmbientLight(glm::vec3 skyColor, float skyIntensity, glm::vec3 groundColor, float groundIntensity)
    {
        ambientLightSkyColor = skyColor;  
        ambientLightSkyIntensity = skyIntensity; 
        ambientLightGroundColor = groundColor; 
        ambientLightGroundIntensity = groundIntensity; 
    }

    void Start(std::vector<float>& positions, 
                std::vector<float>& normals, 
                std::vector<float>& lmuvs, 
                std::vector<unsigned int>& indices, 
                std::vector<unsigned int>& meshes, 
                std::vector<std::string> materials, 
                BVH& bvh, 
                int width, 
                int height, 
                int frameC, 
                int bounces)
    {
        if(!CheckValidContext()) return; 

        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE oldcontext = emscripten_webgl_get_current_context();
        emscripten_webgl_make_context_current(context);

        //
        //Set up variables 
        renderCount = 0;
        currentFrame = 0; 
        frameBufferTextureWidth = width; 
        frameBufferTextureHeight = height; 
        frameCount = frameC; 
        maxBounces = bounces; 
        normalMapHasRun = false; 
        std::cout << "maxBounces: " << maxBounces << "\n"; 

        //
        //Generate Random noise texture 
        randGenerator.seed( std::chrono::system_clock::now().time_since_epoch().count() ); //seed the generator
        fastRand.seed( std::chrono::system_clock::now().time_since_epoch().count() ); 
        GenerateNoiseTexture(); 

        //
        //Create Framebuffers
        glGenTextures(1, &normalFrameBufferTexture); 
        glBindTexture(GL_TEXTURE_2D, normalFrameBufferTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, frameBufferTextureWidth, frameBufferTextureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glGenFramebuffers(1, &frameBufferNormal);
        glBindFramebuffer(GL_FRAMEBUFFER, frameBufferNormal);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, normalFrameBufferTexture, 0);


        glGenTextures(1, &frameBuffer2Texture); 
        glBindTexture(GL_TEXTURE_2D, frameBuffer2Texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, frameBufferTextureWidth, frameBufferTextureHeight, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glGenFramebuffers(1, &frameBuffer2);
        glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer2);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frameBuffer2Texture, 0);


        glGenTextures(1, &frameBufferTexture); 
        glBindTexture(GL_TEXTURE_2D, frameBufferTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, frameBufferTextureWidth, frameBufferTextureHeight, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glGenFramebuffers(1, &frameBuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frameBufferTexture, 0);
 

        //
        //Copy mesh data to buffers for rendering
        glGenBuffers(1, &vbo); 
        glGenBuffers(1, &ebo); 

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        positionSize = positions.size() * sizeof(float); 
        normalSize = normals.size() * sizeof(float); 
        lmuvSize = lmuvs.size() * sizeof(float);
        glBufferData(GL_ARRAY_BUFFER, positionSize + normalSize + lmuvSize, NULL, GL_STATIC_DRAW);

        std::cout << "copy position\n";
        glBufferSubData(GL_ARRAY_BUFFER, 0, positionSize, positions.data()); 
        std::cout << "copy normal\n";
        glBufferSubData(GL_ARRAY_BUFFER, positionSize, normalSize, normals.data());
        std::cout << "copy lmuv\n"; 
        glBufferSubData(GL_ARRAY_BUFFER, positionSize + normalSize, lmuvSize, lmuvs.data()); 
 

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        uint32_t elementSize = indices.size() * sizeof(uint32_t); 
        std::cout << "copy indices\n"; 
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, elementSize, indices.data(), GL_STATIC_DRAW);
       
        indexCount = indices.size(); 
        triangleCount = indexCount / 3; 

        //
        //Construct Scene data 

        //bvh get data
        std::vector<float> nodeBounds;
        std::vector<unsigned int> nodeData; 
        std::vector<unsigned int> primitiveIndices; 
        bvhNodeCount = bvh.GetBVHData(nodeBounds, nodeData, primitiveIndices, false); 

        //build material indicies and material color texture 
        std::vector<unsigned int> materialIndices; 
        std::vector<unsigned char> materialColors;
        materialIndices.reserve(primitiveIndices.size());
        materialColors.reserve(materials.size()/2 * 3);  
        for(int m = 0; m < meshes.size() / 4; m++)
        {
            //convert hex color to bytes, take colors from material col and average texture col 
            std::string colHex = materials[m*2 + 0]; 
            std::string avTexColHex = materials[m*2 + 1]; 
            col8 col = colorFromHexStr(colHex); 
            col8 avTexCol = colorFromHexStr(avTexColHex);
            col8 blendC = blendColsMult(col, avTexCol);

            materialColors.push_back(blendC.r);
            materialColors.push_back(blendC.g); 
            materialColors.push_back(blendC.b); 
            

            //get index to material for triangles 
            unsigned int iOffset = meshes[m * 4 + 0]; 
            unsigned int iLen = meshes[m * 4 + 1]; 
            for(int i = 0; i < iLen/3; i++)
            {
                materialIndices.push_back(m); 
            }
            
        } 

        // Reorder triangles so that the nodes point directly to triangles! for better performance
        std::vector<unsigned int> indicesReorder;
        indicesReorder.reserve(primitiveIndices.size() * 4);
        for(int i = 0; i < primitiveIndices.size(); i++)
        {
            unsigned int triangleIndex = primitiveIndices[i];
            indicesReorder.push_back(indices[triangleIndex * 3 + 0]); 
            indicesReorder.push_back(indices[triangleIndex * 3 + 1]); 
            indicesReorder.push_back(indices[triangleIndex * 3 + 2]); 
            indicesReorder.push_back(materialIndices[triangleIndex]);
        }

        //copy over scene data into textures
        positionsTex = PackTexture3f(positions);
        normalsTex = PackTexture3f(normals);
        trianglesTex = PackTexture4u(indicesReorder);
        bvhBoundsTex = PackTexture3f(nodeBounds);
        bvhNodeDataTex = PackTexture3u(nodeData); 
        materialColorsTex = PackTexture3(materialColors); 

        //
        //Generate square shape 
        glGenBuffers(1, &vboPostProc); 
        glBindBuffer(GL_ARRAY_BUFFER, vboPostProc);
        std::vector<float> squareShape = {0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  
                                          1.0, 1.0,  0.0, 1.0,  0.0, 0.0}; 
        glBufferData(GL_ARRAY_BUFFER, squareShape.size() * sizeof(float), squareShape.data(), GL_STATIC_DRAW);
        

        //
        //Return to origional context
        emscripten_webgl_make_context_current(oldcontext); 
    }

    void RenderLightMapFrames(int framesToRender=1)
    {
        //
        //Startup context
        if(!CheckValidContext()) return; 

        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE oldcontext = emscripten_webgl_get_current_context();
        emscripten_webgl_make_context_current(context);
        
        //
        //Run Render

        glUseProgram(renderProgram);

        glClearColor(0.0, 0.0, 0.0, 0.0); 

        //bind vbo ebo for render
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, (void*)0); 
        glEnableVertexAttribArray(0); 
        glVertexAttribPointer(1, 3, GL_FLOAT, false,  0, (void*)positionSize); 
        glEnableVertexAttribArray(1); 
        glVertexAttribPointer(2, 2, GL_FLOAT, false, 0, (void*)(positionSize + normalSize)); 
        glEnableVertexAttribArray(2);

        //Bind textures 
        glUniform1i( glGetUniformLocation(renderProgram, "positionsTex"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, positionsTex);
        
        glUniform1i( glGetUniformLocation(renderProgram, "normalsTex"), 1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normalsTex); 

        glUniform1i( glGetUniformLocation(renderProgram, "trianglesTex"), 2);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, trianglesTex); 

        glUniform1i( glGetUniformLocation(renderProgram, "bvhBoundsTex"), 3);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, bvhBoundsTex); 

        glUniform1i( glGetUniformLocation(renderProgram, "bvhNodeDataTex"), 4);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, bvhNodeDataTex); 

        glUniform1i( glGetUniformLocation(renderProgram, "materialColorsTex"), 5);
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, materialColorsTex); 

        glUniform1i( glGetUniformLocation(renderProgram, "noiseTex"), 6 );

        glUniform1i( glGetUniformLocation(renderProgram, "accumulationTex"), 7 );


        //set per sample uniform setting in shader 
        glUniform1i( glGetUniformLocation(renderProgram, "maxBounces"), maxBounces);  
        //glUniform1i( glGetUniformLocation(renderProgram, "triangleCount"), triangleCount);//only for brute force algo
        glUniform1i( glGetUniformLocation(renderProgram, "bvhNodeCount"), bvhNodeCount); 
        glUniform1i( glGetUniformLocation(renderProgram, "texBufferWidth"), texBufferWidth); 
        glUniform1i( glGetUniformLocation(renderProgram, "frameBufferWidth"), frameBufferTextureWidth); 
        glUniform1i( glGetUniformLocation(renderProgram, "frameBufferHeight"), frameBufferTextureHeight); 
        glUniform1i( glGetUniformLocation(renderProgram, "currentFrame"), currentFrame); 
        glUniform1i( glGetUniformLocation(renderProgram, "frameCount"), frameCount); 
        glUniform1i( glGetUniformLocation(renderProgram, "randomSamplingMode"), 1); //0: uniform sampling, 1 Reuben Sampling

        glUniform3f( glGetUniformLocation(renderProgram, "mainLightDirection"), mainLightDirection.x, mainLightDirection.y, mainLightDirection.z); 
        glUniform3f( glGetUniformLocation(renderProgram, "mainLightColor"), mainLightColor.x, mainLightColor.y, mainLightColor.z ); 
        glUniform1f( glGetUniformLocation(renderProgram, "mainLightIntensity"), mainLightIntensity); 
        glUniform1f( glGetUniformLocation(renderProgram, "mainLightRadius"), mainLightRadius);

        glUniform3f( glGetUniformLocation(renderProgram, "ambientLightSkyColor"), ambientLightSkyColor.x, ambientLightSkyColor.y, ambientLightSkyColor.z); 
        glUniform1f( glGetUniformLocation(renderProgram, "ambientLightSkyIntensity"), ambientLightSkyIntensity); 
        glUniform3f( glGetUniformLocation(renderProgram, "ambientLightGroundColor"), ambientLightGroundColor.x, ambientLightGroundColor.y, ambientLightGroundColor.z); 
        glUniform1f( glGetUniformLocation(renderProgram, "ambientLightGroundIntensity"), ambientLightGroundIntensity); 

        for(int i = 0; i < framesToRender; i++)
        { 
            //bind noise texture and rebuild
            glActiveTexture(GL_TEXTURE6);
            UpdateNoiseTexture(); //update noise texture along the way 
            glBindTexture(GL_TEXTURE_2D_ARRAY, noiseTex); 

            //bind correct accumulation buffer
            glActiveTexture(GL_TEXTURE7);
            glBindTexture(GL_TEXTURE_2D, renderCount % 2 == 1 ? frameBuffer2Texture : frameBufferTexture); //read from fb 2  if odd

            //bind correct fb
            GLuint fb = renderCount % 2 == 1 ? frameBuffer : frameBuffer2; //Attach frameBuffer1 is odd, if even attach framebuffer 2
            glBindFramebuffer(GL_FRAMEBUFFER, fb);

            //DRAW!
            glClear(GL_COLOR_BUFFER_BIT);
            glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);

            renderCount++;
            currentFrame++; 
        }

        //
        //Swap context back 
        emscripten_webgl_make_context_current(oldcontext);

    }

    void AverageSamples()
    {
        if(!CheckValidContext()) return; 

        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE oldcontext = emscripten_webgl_get_current_context();
        emscripten_webgl_make_context_current(context);

        //
        //Run Accumulate stage
        glUseProgram(accumulateProgram); 
        
        glBindBuffer(GL_ARRAY_BUFFER, vboPostProc);
        glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, (void*)0); 
        glEnableVertexAttribArray(0); 

        glUniform1i( glGetUniformLocation(accumulateProgram, "img"), 0);
        glActiveTexture(GL_TEXTURE0);

        glUniform1i( glGetUniformLocation(accumulateProgram, "imgWidth"), frameBufferTextureWidth); 
        glUniform1i( glGetUniformLocation(accumulateProgram, "imgHeight"), frameBufferTextureHeight); 
        glUniform1i( glGetUniformLocation(accumulateProgram, "sampleCount"), currentFrame); 
        
        GLuint fb = renderCount % 2 == 1 ? frameBuffer : frameBuffer2; 

        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glBindTexture(GL_TEXTURE_2D, renderCount % 2 == 1 ? frameBuffer2Texture : frameBufferTexture);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        renderCount++; 

        //
        //return old context
        emscripten_webgl_make_context_current(oldcontext);
    }

    void RenderNormalMap()
    {
        if(!CheckValidContext()) return; 

        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE oldcontext = emscripten_webgl_get_current_context();
        emscripten_webgl_make_context_current(context);

        //
        //Run  
        glUseProgram(normalMapProgram); 

        //bind vbo ebo for render
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, (void*)0); 
        glEnableVertexAttribArray(0); 
        glVertexAttribPointer(1, 3, GL_FLOAT, false,  0, (void*)positionSize); 
        glEnableVertexAttribArray(1); 
        glVertexAttribPointer(2, 2, GL_FLOAT, false, 0, (void*)(positionSize + normalSize)); 
        glEnableVertexAttribArray(2);

        glBindFramebuffer(GL_FRAMEBUFFER, frameBufferNormal);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
    
        //
        //return old context
        emscripten_webgl_make_context_current(oldcontext);

        normalMapHasRun = true; 
    }

    void Dilate()
    {
        if(!CheckValidContext()) return; 

        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE oldcontext = emscripten_webgl_get_current_context();
        emscripten_webgl_make_context_current(context);

        //
        //Run Dilate
        glUseProgram(postProcDilateProgram); 

        glBindBuffer(GL_ARRAY_BUFFER, vboPostProc);
        glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, (void*)0); 
        glEnableVertexAttribArray(0); 

        glUniform1i( glGetUniformLocation(postProcDilateProgram, "img"), 0);
        glActiveTexture(GL_TEXTURE0);

        glUniform1i( glGetUniformLocation(postProcDilateProgram, "imgWidth"), frameBufferTextureWidth); 
        glUniform1i( glGetUniformLocation(postProcDilateProgram, "imgHeight"), frameBufferTextureHeight); 
        
        GLuint fb = renderCount % 2 == 1 ? frameBuffer : frameBuffer2; 

        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glBindTexture(GL_TEXTURE_2D, renderCount % 2 == 1 ? frameBuffer2Texture : frameBufferTexture);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        renderCount++; 

        //
        //return old context
        emscripten_webgl_make_context_current(oldcontext);
    }

    void RunEdgeAwareFilter()
    {
        //generate noraml maps if needed 
        if(!normalMapHasRun)
            RenderNormalMap(); 


        if(!CheckValidContext()) return; 

        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE oldcontext = emscripten_webgl_get_current_context();
        emscripten_webgl_make_context_current(context);

        //
        //Run  

        glUseProgram(postProcEdgeAwareProgram); 

        glBindBuffer(GL_ARRAY_BUFFER, vboPostProc);
        glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, (void*)0); 
        glEnableVertexAttribArray(0); 

        glUniform1i( glGetUniformLocation(postProcEdgeAwareProgram, "img"), 0);
        glActiveTexture(GL_TEXTURE0);

        glUniform1i( glGetUniformLocation(postProcEdgeAwareProgram, "imgWidth"), frameBufferTextureWidth); 
        glUniform1i( glGetUniformLocation(postProcEdgeAwareProgram, "imgHeight"), frameBufferTextureHeight); 
        
        GLuint fb = renderCount % 2 == 1 ? frameBuffer : frameBuffer2; 

        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glBindTexture(GL_TEXTURE_2D, renderCount % 2 == 1 ? frameBuffer2Texture : frameBufferTexture);

        glUniform1i( glGetUniformLocation(postProcEdgeAwareProgram, "normMap"), 1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normalFrameBufferTexture);

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        renderCount++; 
    
        //
        //return old context
        emscripten_webgl_make_context_current(oldcontext);
    }

    void RunGaussFilter()
    {
        if(!CheckValidContext()) return; 

        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE oldcontext = emscripten_webgl_get_current_context();
        emscripten_webgl_make_context_current(context);

        //
        //Run Gauss 
        glUseProgram(postProcGaussProgram); 

        glBindBuffer(GL_ARRAY_BUFFER, vboPostProc);
        glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, (void*)0); 
        glEnableVertexAttribArray(0); 

        glUniform1i( glGetUniformLocation(postProcGaussProgram, "img"), 0);
        glActiveTexture(GL_TEXTURE0);

        glUniform1i( glGetUniformLocation(postProcGaussProgram, "imgWidth"), frameBufferTextureWidth); 
        glUniform1i( glGetUniformLocation(postProcGaussProgram, "imgHeight"), frameBufferTextureHeight); 
        
        GLuint fb = renderCount % 2 == 1 ? frameBuffer : frameBuffer2; 

        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glBindTexture(GL_TEXTURE_2D, renderCount % 2 == 1 ? frameBuffer2Texture : frameBufferTexture);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        renderCount++; 
    

        //
        //return old context
        emscripten_webgl_make_context_current(oldcontext);
    }

    void RunMedianFilter()
    {
        if(!CheckValidContext()) return; 

        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE oldcontext = emscripten_webgl_get_current_context();
        emscripten_webgl_make_context_current(context);
        
        //
        //Run Median

        glUseProgram(postProcMedianProgram);

        glBindBuffer(GL_ARRAY_BUFFER, vboPostProc);
        glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, (void*)0); 
        glEnableVertexAttribArray(0); 
        
        glUniform1i( glGetUniformLocation(postProcMedianProgram, "img"), 0);
        glActiveTexture(GL_TEXTURE0);

        glUniform1i( glGetUniformLocation(postProcMedianProgram, "imgWidth"), frameBufferTextureWidth); 
        glUniform1i( glGetUniformLocation(postProcMedianProgram, "imgHeight"), frameBufferTextureHeight); 
        
        GLuint fb = renderCount % 2 == 1 ? frameBuffer : frameBuffer2; 

        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glBindTexture(GL_TEXTURE_2D, renderCount % 2 == 1 ? frameBuffer2Texture : frameBufferTexture);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        renderCount++; 
       
        //
        //return old context
        emscripten_webgl_make_context_current(oldcontext);
    }

    void RenderToCanvas()
    {
        if(!CheckValidContext()) return; 

        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE oldcontext = emscripten_webgl_get_current_context();
        emscripten_webgl_make_context_current(context);

        //
        //Pass through shader  (render back to canvas )

        glUseProgram(postProcPassThroughProgram);

        glBindBuffer(GL_ARRAY_BUFFER, vboPostProc);
        glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, (void*)0); 
        glEnableVertexAttribArray(0); 
        
        glUniform1i( glGetUniformLocation(postProcPassThroughProgram, "img"), 0);
        glActiveTexture(GL_TEXTURE0);

        glUniform1i( glGetUniformLocation(postProcPassThroughProgram, "imgWidth"), frameBufferTextureWidth);
        glUniform1i( glGetUniformLocation(postProcPassThroughProgram, "imgHeight"), frameBufferTextureHeight);

        glBindFramebuffer(GL_FRAMEBUFFER, NULL);

        glBindTexture(GL_TEXTURE_2D, renderCount % 2 == 1 ? frameBuffer2Texture : frameBufferTexture);

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        //
        //return old context
        emscripten_webgl_make_context_current(oldcontext);
    }

    void AverageSamplesAndRenderToCanvas() //This is to get a snapshot of the render at any point during the render phase
    {
        if(!CheckValidContext()) return; 

        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE oldcontext = emscripten_webgl_get_current_context();
        emscripten_webgl_make_context_current(context);

        //
        //Pass through shader  (render back to canvas )

        glUseProgram(accumulateProgram);

        glBindBuffer(GL_ARRAY_BUFFER, vboPostProc);
        glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, (void*)0); 
        glEnableVertexAttribArray(0); 
        
        glUniform1i( glGetUniformLocation(accumulateProgram, "img"), 0);
        glActiveTexture(GL_TEXTURE0);

        glUniform1i( glGetUniformLocation(accumulateProgram, "imgWidth"), frameBufferTextureWidth);
        glUniform1i( glGetUniformLocation(accumulateProgram, "imgHeight"), frameBufferTextureHeight);
        glUniform1i( glGetUniformLocation(accumulateProgram, "sampleCount"), currentFrame); 

        glBindFramebuffer(GL_FRAMEBUFFER, NULL);

        glBindTexture(GL_TEXTURE_2D, renderCount % 2 == 1 ? frameBuffer2Texture : frameBufferTexture);

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        //
        //return old context
        emscripten_webgl_make_context_current(oldcontext);
    }

    void Stop()
    {
        if(!CheckValidContext()) return; 
        
        EMSCRIPTEN_WEBGL_CONTEXT_HANDLE oldcontext = emscripten_webgl_get_current_context();
        emscripten_webgl_make_context_current(context);

        //
        //Clean up 

        //(The gl functions silently ignore any calls to input 0!)
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &ebo);

        glDeleteBuffers(1, &vboPostProc); 

        glDeleteTextures(1, &positionsTex); 
        glDeleteTextures(1, &normalsTex); 
        glDeleteTextures(1, &trianglesTex); 
        glDeleteTextures(1, &bvhBoundsTex);
        glDeleteTextures(1, &bvhNodeDataTex);
        glDeleteTextures(1, &noiseTex);
        glDeleteTextures(1, &materialColorsTex); 

        glDeleteFramebuffers(1, &frameBuffer);
        glDeleteTextures(1, &frameBufferTexture); 
        glDeleteFramebuffers(1, &frameBuffer2);
        glDeleteTextures(1, &frameBuffer2Texture);

        //
        //return old context
        emscripten_webgl_make_context_current(oldcontext);
    }
};