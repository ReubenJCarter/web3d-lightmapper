import { Engine } from "@babylonjs/core/Engines/engine";
import { Scene } from "@babylonjs/core/scene";
import { Color3, Vector3 } from "@babylonjs/core/Maths/math";
import { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { Mesh, MeshBuilder, StandardMaterial, Texture, VertexData } from "@babylonjs/core";

import "@babylonjs/core/Materials/standardMaterial";

import LightMapper from "./build/lightmapperBabylon";

let consoleText = ``; 
function Log(str, type='message'){
    let col = type == 'message' ? '#FFFFFF' : '#FF5555'; 
    console.log(str); 
    let el = document.getElementById("console"); 
    consoleText += '<span style="color:'+col+'; ">' + str + '</span>' + '<br>'; 
    el.innerHTML = consoleText; 
}

window.onload = ()=>{

    console.log('Tests Starting'); 
    
    //Scene
    const canvas = document.getElementById("main-canvas");
    const engine = new Engine(canvas);
    var scene = new Scene(engine);

    //Camera
    var camera = new FreeCamera("camera1", new Vector3(0, 5, -10), scene);
    camera.setTarget(Vector3.Zero());
    camera.attachControl(canvas, true);

    //Light 
    var light = new HemisphericLight("light1", new Vector3(-0.5, 1.0, -0.5), scene);
    light.intensity = 1;

    //box
    let box0 = MeshBuilder.CreateBox('box0', {size:1}, scene)
    box0.position.y = 1; 
    box0.position.x = 1;
    
    box0.material = new StandardMaterial('box0_material', scene); 
    box0.material.diffuseColor = new Color3(1.0, 0.0, 0.0); 

    let box1 = MeshBuilder.CreateBox('box1', {size:1}, scene)
    box1.position.y = 1; 
    box1.position.x = -1;
    
    box1.material = new StandardMaterial('box1_material', scene); 
    box1.material.diffuseColor = new Color3(0.0, 1.0, 0.0); 

    let box2 = MeshBuilder.CreateBox('box2', {size:1}, scene)
    box2.position.z = -1;
    box2.position.y = 0; 
    box2.position.x = 1;

    box2.material = new StandardMaterial('box2_material', scene); 
    box2.material.diffuseColor = new Color3(0.0, 0.0, 1.0); 

    //Ground
    let ground = MeshBuilder.CreateGround("ground1", {width:6, height:6, subdivisions:2}, scene);

    let lightMapper = new LightMapper(); 
    let lightData = {
        main:{
            direction:new Vector3(0.5, -1.0, 0.5), 
            color:new Color3(1, 1, 1), 
            power: 0.14,
            radius:0.2, 
        }, 
        ambient: {
            skyColor: new Color3(1, 1, 1), 
            skyIntensity: 0.36, 
            groundColor: new Color3(1, 1, 1), 
            groundIntensity: 0.2,
        }, 
    }
    

    //Run
    engine.runRenderLoop(() => {
        scene.render();
    });

    Log('Running Light Mapper');
  
    lightMapper.run(scene, [ground, box0, box1, box2], 2048, 2048, 500, lightData, (progress)=>{
        console.log(progress);
        Log(`${Math.round(progress*100)}% Complete`);
    }).then((lmData)=>{

        Log(`Light Map Done`);
        console.log(lmData); 
        //Go over each mesh group and create  a new mesh for it
        for(let i = 0; i < lmData.sceneData.length; i++){
            let bakedData = lmData.sceneData[i]; 

            for(let m = 0; m < bakedData.meshes.length / 4; m++){
                let iOff = bakedData.meshes[m * 4 + 0]
                let iCount = bakedData.meshes[m * 4 + 1]; 
                let vOff = bakedData.meshes[m * 4 + 2];
                let vCount = bakedData.meshes[m * 4 + 3];
                let mat = bakedData.materials[m*2+0]; 

                let bakedSceneMesh = new Mesh("bakedSceneMesh_"+i, scene);
                let vertexData = new VertexData(); 

                //update vertex data and make mesh 
                vertexData.colors = bakedData.colors.slice(4 * vOff, 4 * (vOff+vCount)); 
                vertexData.positions = bakedData.positions.slice(3 * vOff, 3 * (vOff+vCount));
                vertexData.normals  = bakedData.normals.slice(3 * vOff, 3 * (vOff+vCount));
                vertexData.uvs = bakedData.lmuvs.slice(2 * vOff, 2 * (vOff+vCount));
                vertexData.uvs2 = bakedData.lmuvs.slice(2 * vOff, 2 * (vOff+vCount));
                vertexData.indices = bakedData.indices.slice(iOff, iOff + iCount).map(inx=>inx-vOff);
                vertexData.applyToMesh(bakedSceneMesh); 

                let material = new StandardMaterial(scene); 
                material.useLightmapAsShadowmap = true; 
                material.lightmapTexture = new Texture(lmData.lightMap, scene, true); 
                material.diffuseColor = Color3.FromHexString(mat); 
                bakedSceneMesh.material = material;
            }
        }

        ground.isVisible = false; 
        box0.isVisible = false; 
        box1.isVisible = false; 
        box2.isVisible = false; 
        
        //scene.removeLight(light);

    }); 

    
}