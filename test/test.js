import { Engine } from "@babylonjs/core/Engines/engine";
import { Scene } from "@babylonjs/core/scene";
import { Color3, Vector3 } from "@babylonjs/core/Maths/math";
import { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { DirectionalLight, Light, Mesh, MeshBuilder, SceneLoader, StandardMaterial, Texture, VertexData } from "@babylonjs/core";

import "@babylonjs/core/Materials/standardMaterial";
import "@babylonjs/loaders/glTF";

import { LightMapper, addBakedMeshesToScene } from "..";

let consoleText = ``; 
function Log(str, type='message'){
    let col = type == 'message' ? '#FFFFFF' : '#FF5555'; 
    console.log(str); 
    let el = document.getElementById("console"); 
    consoleText += '<span style="color:'+col+'; ">' + str + '</span>' + '<br>'; 
    el.innerHTML = consoleText; 
    el.scrollTop = el.scrollHeight;
}

window.onload = async ()=>{

    Log('Tests Starting'); 
    
    //Scene
    const canvas = document.getElementById("main-canvas");
    const engine = new Engine(canvas);
    var scene = new Scene(engine);

    //Camera
    var camera = new FreeCamera("camera1", new Vector3(0, 1, 3.5), scene);
    camera.setTarget(new Vector3(0, 1, 0));
    camera.attachControl(canvas, true);

    //Light 
    var light = new DirectionalLight("light1", new Vector3(0, -1, -1), scene);
    light.intensity = 1;
    const ambientLight = new HemisphericLight("HemiLight", new Vector3(0, 1, 0), scene);

    

    //Cornell 
    await SceneLoader.ImportMeshAsync('', './models/', 'scene.gltf', scene); 

/*
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
*/
    //Run light mapper
    engine.runRenderLoop(() => {
        scene.render();
    });


    //Light mapper settings
    let lightMapper = new LightMapper(); 
    let lightData = {
        main:{
            direction: light.direction.scale(-1), 
            color: new Color3(1, 1, 1), 
            power: 0.2,
            radius: 0.2, 
        }, 
        ambient: {
            skyColor: new Color3(1, 1, 1), 
            skyIntensity: 0.5, 
            groundColor: new Color3(1, 1, 1), 
            groundIntensity: 0.2,
        }, 
    }

    Log('Running Light Mapper');

    let allMeshes = []; 
    for(let mesh of scene.meshes){
        allMeshes.push(mesh); 
    }
  
    lightMapper.run(scene, allMeshes, 2048, 2048, 500, 3, lightData, {
        gaussFilterCount:10,
    }, (progress)=>{
        Log(`${Math.round(progress*100)}% Complete`);
    }, ).then((lmData)=>{

        Log(`Light Map Done`);
        console.log(lmData); 

        addBakedMeshesToScene(lmData, scene); 

        for(let oldMesh of allMeshes){
            oldMesh.isVisible = false; 
        }

        light.lightmapMode = Light.LIGHTMAP_SPECULAR;
        ambientLight.lightmapMode = Light.LIGHTMAP_SPECULAR; 
    }); 

    
}