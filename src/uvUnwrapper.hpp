#pragma once

#define XA_MULTITHREADED 0
#include "thirdparty/xatlas/xatlas.h"
#include "thirdparty/xatlas/xatlas.cpp"

class UVUnwrapper
{
public: 
    UVUnwrapper()
    {

    }

    void Run(std::vector<float>& positions, 
            std::vector<float>& normals, 
            std::vector<float>& uvs, 
            std::vector<float>& lmuvs,
            std::vector<unsigned int>& indices, 
            std::vector<unsigned int>& meshes)
    {
        std::cout << "Calculating light map UVs\n";

        //
        //Build atlases by adding all meshes 
        xatlas::Atlas* atlas = xatlas::Create();

        for(int m = 0; m < meshes.size()/4; m++)
        {
            //std::cout << "adding mesh " << m << "\n"; 

            //get mesh index offset and count 
            uint32_t iOffset = meshes[m * 4 + 0];
            uint32_t iCount = meshes[m * 4 + 1]; 
            uint32_t vOffset = meshes[m * 4 + 2]; 
            uint32_t vCount = meshes[m * 4 + 3];       

            //put data into xatlas mesh object 
            xatlas::MeshDecl meshDecl;
            meshDecl.vertexCount = vCount;
            
            meshDecl.vertexPositionData = &(positions[vOffset * 3]);
            meshDecl.vertexPositionStride = sizeof(float) * 3;
            
            meshDecl.vertexNormalData = &(normals[vOffset * 3]);
            meshDecl.vertexNormalStride = sizeof(float) * 3;
            
            meshDecl.indexCount = iCount;
            meshDecl.indexData = &(indices[iOffset]);
            meshDecl.indexOffset = -vOffset; 
            meshDecl.indexFormat = xatlas::IndexFormat::UInt32;

            xatlas::AddMeshError error = xatlas::AddMesh(atlas, meshDecl, meshes.size()); //mesh data is copied...

            if (error != xatlas::AddMeshError::Success) 
            {
                xatlas::Destroy(atlas);
                std::cout << "UVUnwrapper Error adding mesh " << m << "  " << xatlas::StringForEnum(error) << "\n";
                return ;
            }
        }


        //
        //Do Packing
        std::cout << "Running Generate\n"; 
        //xatlas::Generate(atlas);
        xatlas::ChartOptions chartOptions; 
        std::cout << "Computing Charts\n"; 
        xatlas::ComputeCharts(atlas); 
        xatlas::PackOptions packOptions; 
        packOptions.padding = 4; 
        std::cout << "Packing Charts\n"; 
        xatlas::PackCharts(atlas, packOptions);

        //
        //Rebuild all mesh data after pack
        std::cout << "Copying new data\n"; 

        std::vector<float> newPositions; 
        std::vector<float> newNormals; 
        std::vector<float> newUvs; 
        lmuvs.clear(); 

        uint32_t vCount = 0; 
        for(int m = 0; m < meshes.size()/4; m++)
        {
            uint32_t xVCount = atlas->meshes[m].vertexCount; 
            xatlas::Vertex* xVerts = atlas->meshes[m].vertexArray; 
            uint32_t* xIndices = atlas->meshes[m].indexArray; 
            int atlasWidth = atlas->width;
            int atlasHeight = atlas->height;

            //add the position norals and uv lmuvs to the new vectors using the index from the xatlas verticies
            uint32_t oldVOffset = meshes[m*4 + 2];
            for(uint32_t i = 0; i < xVCount; i++)
            {
                uint32_t inx = (oldVOffset + xVerts[i].xref) * 3; 
                uint32_t inxuv = (oldVOffset + xVerts[i].xref) * 2;
                newPositions.push_back( positions[inx] ); 
                newPositions.push_back( positions[inx + 1] ); 
                newPositions.push_back( positions[inx + 2] ); 
                newNormals.push_back( normals[inx] ); 
                newNormals.push_back( normals[inx + 1] ); 
                newNormals.push_back( normals[inx + 2] ); 
                newUvs.push_back( uvs[inxuv] ); 
                newUvs.push_back( uvs[inxuv + 1] ); 
                lmuvs.push_back( xVerts[i].uv[0] / atlasWidth); 
                lmuvs.push_back( xVerts[i].uv[1] / atlasHeight); 
            }

            //add new indecies to old vector
            uint32_t iOffset = meshes[m*4 + 0]; 
            uint32_t iCount = meshes[m*4 + 1]; 
            for(uint32_t i = 0; i < iCount; i++)
            {
                indices[iOffset + i] = vCount + xIndices[i]; 
            }

            //update meshes vectors 
            meshes[m*4 + 2] = vCount;
            meshes[m*4 + 3] = xVCount;

            vCount += xVCount;
        }

        //replace the origional vertex vectors 
        positions.clear(); 
        normals.clear(); 
        uvs.clear(); 

        //positions.reserve(newPositions.size());
        //normals.reserve(newNormals.size());
        //uvs.reserve(newUvs.size());

        for(uint32_t i = 0; i < newPositions.size(); i++)
            positions.push_back(newPositions[i]); 
        for(uint32_t i = 0; i < newNormals.size(); i++)
            normals.push_back(newNormals[i]); 
        for(uint32_t i = 0; i < newUvs.size(); i++)
            uvs.push_back(newUvs[i]); 

        //Cleanup 
        xatlas::Destroy(atlas);


        std::cout << "Finished, vert count:" << positions.size() / 3 << "\n"; 
    }
};