#pragma once

#include <sstream>
#include <functional>

#include <bvh/bvh.hpp>
#include <bvh/vector.hpp>
#include <bvh/triangle.hpp>
#include <bvh/sphere.hpp>
#include <bvh/ray.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/node_layout_optimizer.hpp>


class BVH
{
public:
    using Scalar      = float;
    using Vector3     = bvh::Vector3<Scalar>;
    using Triangle    = bvh::Triangle<Scalar>;
    using BoundingBox = bvh::BoundingBox<Scalar>;
    using Bvh         = bvh::Bvh<Scalar>;

    Bvh bvh;
    unsigned int triangleCount; 

    BVH()
    {

    }

    void Print()
    {
        std::stringstream ss; 

        for(int i = 0; i < bvh.node_count; i++)
        {
            ss << "Node " << i << ", "; 
            int primitive_count = bvh.nodes[i].primitive_count; 
            int first_child_or_primitive = bvh.nodes[i].first_child_or_primitive; 
            ss << "primitive_count " << primitive_count << " first_child_or_primitive " << first_child_or_primitive; 
            ss << "\n"; 
        }

        std::cout << ss.str(); 
    }

    void Construct(std::vector<float>& positions, std::vector<unsigned int>& indicies)
    {
        std::cout << "Constructing BVH\n"; 
        // Create an array of Primitives for the BVH 
        std::vector<Triangle> triangles;
        for(int i = 0; i < indicies.size()/3; i++)
        {
            unsigned int i0 = indicies[i*3+0]; 
            unsigned int i1 = indicies[i*3+1]; 
            unsigned int i2 = indicies[i*3+2]; 

            Vector3 v0 = Vector3(positions[i0*3+0], positions[i0*3+1], positions[i0*3+2]); 
            Vector3 v1 = Vector3(positions[i1*3+0], positions[i1*3+1], positions[i1*3+2]); 
            Vector3 v2 = Vector3(positions[i2*3+0], positions[i2*3+1], positions[i2*3+2]); 

            triangles.push_back( Triangle(v0, v1, v2) );
        }

        //calculate bbs and centers 
        triangleCount = triangles.size(); 
        auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
        auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), triangles.size());

        // run the bvh builder
        bvh::SweepSahBuilder< Bvh > builder(bvh);
        builder.build(global_bbox, bboxes.get(), centers.get(), triangles.size());

        //apply optimization to memory layout of bvh
        bvh::NodeLayoutOptimizer layout_optimizer(bvh);
        layout_optimizer.optimize();
    }

    unsigned int GetBVHData(std::vector<float>& nodeBounds, std::vector<unsigned int>& nodeData, std::vector<unsigned int>& primitiveIndices, bool depthFirstLayout=false)
    {
        nodeBounds.reserve(6 * bvh.node_count); 
        nodeData.reserve(3 * bvh.node_count); 

        //DEPTH FIRST LAYOUT NODES
        if(depthFirstLayout)
        {
            std::function<int(int, int)> depthFirstLayoutRecurse; 
            depthFirstLayoutRecurse = [&](int nodeInx, int count)
            {
                //count is the number of nodes in the array 
                int newCount = count; 

                //
                //write node data in depth first order
                //the first child is always just the next node. so do count +1 for that
                unsigned int first_child_or_primitive = bvh.nodes[nodeInx].primitive_count == 0 ? count+1 : bvh.nodes[nodeInx].first_child_or_primitive; 
               
                nodeData.push_back(first_child_or_primitive); 
                nodeData.push_back(bvh.nodes[nodeInx].primitive_count); 
                nodeData.push_back(1234567890); //Parent place holder 

                //min
                nodeBounds.push_back(bvh.nodes[nodeInx].bounds[0]); 
                nodeBounds.push_back(bvh.nodes[nodeInx].bounds[2]);
                nodeBounds.push_back(bvh.nodes[nodeInx].bounds[4]); 

                //max
                nodeBounds.push_back(bvh.nodes[nodeInx].bounds[1]); 
                nodeBounds.push_back(bvh.nodes[nodeInx].bounds[3]); 
                nodeBounds.push_back(bvh.nodes[nodeInx].bounds[5]); 

                newCount++;

                //
                //recurse on chhldren (left first)
                bool hasChildren = bvh.nodes[nodeInx].primitive_count == 0; 
                
                if(hasChildren)
                {
                    int firstChild = bvh.nodes[nodeInx].first_child_or_primitive; 
                    newCount = depthFirstLayoutRecurse(firstChild, newCount); 
                    newCount = depthFirstLayoutRecurse(firstChild+1, newCount); 
                }

                //set the skip index as newCount
                nodeData[count * 3 + 2] = newCount; 
                
                //return the count of the number of nodes in this branch 
                return newCount; 
            }; 
            depthFirstLayoutRecurse(0, 0);

        }

        //NON DEPTH FIRST LAYOUT
        else
        {
            for(int i = 0; i < bvh.node_count; i++)
            {
                nodeData.push_back(bvh.nodes[i].first_child_or_primitive); 
                nodeData.push_back(bvh.nodes[i].primitive_count); 
                nodeData.push_back(1234567890); //Parent place holder (massive number)

                //min
                nodeBounds.push_back(bvh.nodes[i].bounds[0]); 
                nodeBounds.push_back(bvh.nodes[i].bounds[2]);
                nodeBounds.push_back(bvh.nodes[i].bounds[4]); 

                //max
                nodeBounds.push_back(bvh.nodes[i].bounds[1]); 
                nodeBounds.push_back(bvh.nodes[i].bounds[3]); 
                nodeBounds.push_back(bvh.nodes[i].bounds[5]); 
            }

            //compute branch skip indicies (if the skip is to the end of the tree, index will be 0)
            std::function<void(int)> traverseSetBranchSkip; 
            traverseSetBranchSkip = [&](int nodeInx)
            {
                bool hasChildren = nodeData[3*nodeInx + 1] == 0; 
                if(hasChildren)
                {
                    int firstChild = nodeData[3*nodeInx]; 

                    //Set the skip index of the first child to the next child (sibling)! 
                    nodeData[3*firstChild + 2] = firstChild+1;

                    traverseSetBranchSkip(firstChild); 

                    //Set the index of the second child to the parents skip index 
                    unsigned int parentBranchSkipInx = nodeData[3*nodeInx + 2]; 
                    nodeData[3*(firstChild+1) + 2] = parentBranchSkipInx;
                    
                    traverseSetBranchSkip(firstChild+1); 
                }
            }; 
            traverseSetBranchSkip(0); 
        }



        primitiveIndices.reserve(triangleCount); 
        for(int i = 0; i < triangleCount; i++)
        {
            primitiveIndices.push_back( bvh.primitive_indices[i] );
        }

        
       
        
        


        //compute parent node indicies 
        /*
        std::function<void(int, int)> traverseSetParent; 
        traverseSetParent = [&](int nodeInx, int parentInx)
        {
            nodeData[3*nodeInx + 2] = parentInx; 
            bool hasChildren = nodeData[3*nodeInx + 1] == 0; 
            if(hasChildren)
            {
                int firstChild = nodeData[3*nodeInx]; 

                traverseSetParent(firstChild, nodeInx); 

                traverseSetParent(firstChild+1, nodeInx); 
            }
        }; 
        traverseSetParent(0, -1); 
        */


        return bvh.node_count; 
    }

}; 