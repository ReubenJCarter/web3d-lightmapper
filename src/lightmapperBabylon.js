import LightMapperWASM from './lightmapperWasm.mjs';

import { Color3, VertexBuffer, Scene, FreeCamera, Camera, MeshBuilder, Vector3, StandardMaterial, Tools } from "@babylonjs/core";


//Attaches to a scene, creates a new independent scene with the same engine, and uses it to render any texture, then reduces to 1 px so a color
class TexAvColSystem {

	constructor(scene, ) {

		// get the average color of a texture 
		this.engine = scene.getEngine();
		this.avTexColScene = new Scene(this.engine);
		this.avTexColCam = new FreeCamera("avTexColCam", new Vector3(0, 0, -10), this.avTexColScene);
    	this.avTexColCam.mode = Camera.ORTHOGRAPHIC_CAMERA;
		this.avTexColCam.orthoTop = 1;
		this.avTexColCam.orthoBottom = -1;
    	this.avTexColCam.orthoLeft = -1;
    	this.avTexColCam.orthoRight = 1;
		this.avTexColMesh = MeshBuilder.CreatePlane("avTexColMesh",{size:2}, this.avTexColScene);
		this.avTexColMesh.material = new StandardMaterial("avTexColMeshMat", this.avTexColScene); 
		this.avTexColMesh.material.emissiveColor = new Color3(0, 0, 0); 
		this.avTexColMesh.material.diffuseColor = new Color3(0, 0, 0); 
		this.avTexColMesh.material.specularColor = new Color3(0, 0, 0); 
		this.avTexColMesh.material.ambientColor = new Color3(0, 0, 0); 
		this.avTexColMesh.material.backFaceCulling = false; 
		this.avTexColScene.render();

		this.avTexCanvas = document.createElement('canvas');
		this.avTexCanvas.width = 1;
		this.avTexCanvas.height = 1;
	}

	run(texture) {
		if(!texture)
			return '#FFFFFF';

		
		this.avTexColMesh.material.emissiveTexture = texture; 
		this.avTexColScene.render();
		this.avTexColScene.render();
		return Tools.CreateScreenshotUsingRenderTargetAsync(this.engine, this.avTexColCam, 32)
		.then((imgd)=>{
			let img = new Image();
			return new Promise((resolve, reject)=>{ 
				img.onload= ()=>{
					let ctx = this.avTexCanvas.getContext('2d');
					ctx.drawImage(img, 0, 0, 1, 1);
					let c = ctx.getImageData(0, 0, 1, 1).data; 
					let col = new Color3(c[0] / 255, c[1] / 255, c[2] / 255); 
					let hex = col.toHexString(); 
					resolve(hex);
				}
				img.onerror = ()=>{
					reject({message:'failed to load image'}); 
				}
				img.src = imgd;
			}); 
		});
	}

	destroy(){
		this.avTexColScene.dispose(); 
	}
}

/*
Compute lighting uvs for each mesh 
Combine and generate BVHs for all meshes
BVH uploaded to GPU via texture
Render combines mesh
vertex shader maps position from lighting uvs, creates varyings from position and normal 
fragment shader creates ray from po and norm varying 
frag shader path traces ray into bvh texture with N bounces 
writes out color and render to texture used
color texture combined using averaging into accumulator texture. 
Render repeted N times
Denoiser runs on accumulator texture

*/

export default class LightMapper {
	constructor(){
		this.cancel = false;
		this.progress = 0.0; 
	}

	async loadWasm(){  
		try {

			if(!this.wasm){
				let config = {
					locateFile:(path, prefix)=>{
						return path; //just return the path part so the file is loaded from root
					}
				}; 
				this.wasm = await LightMapperWASM(config);
				//this.wasm.TestFunction();
			}

		} catch(err) { 
			console.error(`Unexpected error in LightMapper.loadWasm. [Message: ${err.message}]`);
			throw {message: `Unexpected error in LightMapper.loadWasm. [Message: ${err.message}]`}; 
		}
	};

	async run(scene, meshes, mapWidth, mapHeight, sampleCount, lightData, progressChanged, denoiserInputSamples=50){

		let deNoiseTrainMode = false; 
		
		//clear the cancel 
		this.cancel = false;
		this.progress = 0;

		//Create canvas and context
		let canvas = document.createElement('canvas');
		canvas.width = mapWidth;
		canvas.height = mapHeight;
		canvas.id = 'lmglCanvas';
		const context = canvas.getContext("webgl2");  
		this.canvas = canvas;    
		document.body.appendChild(canvas);
		canvas.style.visibility = 'hidden';
		canvas.style.display = 'none'; 
		
		try{
			await this.loadWasm(); 
		}
		catch(e){

			//Clean up   
			this.canvas.remove(); 
			this.canvas = null;

			throw e; 
		}

		//create new wasm light mapper 
		this.wasmLightMapper = new this.wasm.LightMapper(); 
		let inited = this.wasmLightMapper.Init(this.canvas.id);
		if(!inited){
			throw {message: 'Light mapper could not init'};
		}


		//start timer
		const t0 = performance.now();

		//Create scene data 
		let vecP = new this.wasm.VectorFloat();
		let vecN = new this.wasm.VectorFloat();  
		let vecUV = new this.wasm.VectorFloat();
		let vecLMUV = new this.wasm.VectorFloat(); 
		let vecI = new this.wasm.VectorUInt();
		let vecM = new this.wasm.VectorUInt(); //mesh index (and length)
		let vecT = new this.wasm.VectorFloat(); //mesh transform
		let vecMaterial = new this.wasm.VectorString();
		let vecMInfo = new this.wasm.VectorString(); 

		let deleteSceneVecs = () =>{
			vecP.delete(); 
			vecN.delete();
			vecUV.delete();
			vecLMUV.delete();
			vecI.delete();
			vecM.delete();
			vecT.delete(); 
			vecMaterial.delete(); 
			vecMInfo.delete(); 
		}

		let vCount = 0; 
		let iCount = 0;  

		let internalTextureUIDs = []; 

		// create system to get the average color of a texture 
		let texAvColSystem = new TexAvColSystem(scene); 

		//Function to add a mesh to the scene data
		let addMeshToBlob = (mesh, uid, type, meshMetaData, meshColor='', internalTexture=null,  meshAvTextureColor='') =>{
			let positions = mesh.getVerticesData(VertexBuffer.PositionKind); 
			let normals = mesh.getVerticesData(VertexBuffer.NormalKind); 
			let uvs = mesh.getVerticesData(VertexBuffer.UVKind); 
			let indicies = mesh.getIndices(); 

			if(!positions || !normals || !indicies)
				return ; 
			 
			for(let i = 0; i < positions.length; i++){
				vecP.push_back(positions[i]);
			}
			
			for(let i = 0; i < normals.length; i++){
				vecN.push_back(normals[i]);
			}
			
			if(uvs){
				for(let i = 0; i < uvs.length; i++){ 
					vecUV.push_back(uvs[i]);
				}
			}
			else {
				for(let i = 0; i < positions.length/3; i++){//TODO dot really need uvs or meshes without SAVE SPACE 
					vecUV.push_back(0);
					vecUV.push_back(0);
				}
			}

			for(let i = 0; i < positions.length/3; i++){
				vecLMUV.push_back(0); 
				vecLMUV.push_back(0);
			}

			for(let i = 0; i < indicies.length; i++){
				vecI.push_back(indicies[i] + vCount);
			}

			vecM.push_back(iCount);//offset
			vecM.push_back(indicies.length);//length indices
			vecM.push_back(vCount); //vertex offset 
			vecM.push_back(positions.length/3); //vertex Count 

			let matrix = mesh.computeWorldMatrix(true).m;  
			for(let i = 0; i < matrix.length; i++){
				vecT.push_back(matrix[i]); 
			}

		
			vecMaterial.push_back(meshColor);
			vecMaterial.push_back(meshAvTextureColor);

			vecMInfo.push_back(mesh.name);
			vecMInfo.push_back(uid); //uid of the object the mesh is atached to
			vecMInfo.push_back(type); 
			vecMInfo.push_back(meshMetaData); 
			
			internalTextureUIDs.push(internalTexture); 
			vCount += positions.length /3;
			iCount += indicies.length;
		}
 
	
		for(let mesh of meshes){

			if(mesh.metadata && mesh.metadata.dontInclude)
				continue; 

			let col = mesh.material && mesh.material.diffuseColor ? mesh.material.diffuseColor : Color3.White();
			let colStr = col.toHexString();  
			let internalTexture =  mesh.material && mesh.material.diffuseTexture ? mesh.material.diffuseTexture : null; 
			let avTexCol = await texAvColSystem.run(internalTexture);
			
			let metaData = {}; 
			if(mesh.metadata && mesh.metadata.unique)
				metaData.unique = true;
			if(mesh.metadata && mesh.metadata.invisible)
				metaData.invisible = true; 
			addMeshToBlob(mesh, 'static_scene_data', 'object', JSON.stringify(metaData), colStr, internalTexture, avTexCol); 
		}; 
			

		//dispose of the scene used to create average texture colors 
		texAvColSystem.destroy(); 

		//zero geometry present 
		if(vecI.size() == 0){ 

			deleteSceneVecs(); 

			this.canvas.remove(); 
			this.canvas = null; 
			
			this.wasmLightMapper.delete(); 

			throw {message: 'zero indicies'}
		}

		//
		//render the light maps (inplace modifies all scene data for light mapping, and generate lm texture for scene)
		let renderCount = sampleCount;  
		let framesPerRender = 1;
		let dillateCount = 20; 

		let gaussFilterCount = 0;
		let edgeAwareFilterCount = 0;
		let medianFilterCount = 0; 


		let averageFrameRenderTime = 0;
		let pomiseChain = Promise.resolve(); 

		let addFunctionToChain = (func, timeoutTime=10) =>{
			pomiseChain = pomiseChain.then((promiseInput)=>{ 
 
				return new Promise((resolve, reject)=>{
 
					setTimeout(()=>{
						if(this.cancel){
							this.cancel = false; 
							reject(); 
							return; 
						}
						let ret = func(promiseInput);
						resolve(ret);
					}, timeoutTime);

				}); 

			});
		}  
		
		addFunctionToChain(()=>{
			console.log("Starting Light Mapper");
			let frameCount = framesPerRender * renderCount;
			this.wasmLightMapper.Start(vecP, vecN, vecUV, vecLMUV, vecI, vecM, vecT, vecMaterial, this.canvas.width, this.canvas.height, frameCount, 2);
			
			this.wasmLightMapper.SetMainLight(
				lightData.main.direction.x, lightData.main.direction.y, lightData.main.direction.z, 
				lightData.main.color.r, lightData.main.color.g, lightData.main.color.b,  
				lightData.main.power, lightData.main.radius 
			);

			this.wasmLightMapper.SetAmbientLight(
				lightData.ambient.skyColor.r, lightData.ambient.skyColor.g, lightData.ambient.skyColor.b, lightData.ambient.skyIntensity,
				lightData.ambient.groundColor.r, lightData.ambient.groundColor.g, lightData.ambient.groundColor.b, lightData.ambient.groundIntensity
			); 
			averageFrameRenderTime = performance.now(); 
		}); 

		for(let i = 0; i < renderCount; i++){ 
			addFunctionToChain(()=>{
				
				let tRS = performance.now();                 
				
				this.wasmLightMapper.RenderLightMapFrames(framesPerRender);
				
				const tRE = performance.now();
				console.log("frame " + i ); 

				let frameCount = framesPerRender * renderCount; 
				this.progress = i / frameCount; 
				
				if(progressChanged)
					progressChanged(this.progress); 
			}, 1); 
			
			//if getting denoising training data take a snapshot at correct frame!
			if(deNoiseTrainMode && i == denoiserInputSamples-1){
				addFunctionToChain(()=>{
					console.log("Rendering Training cllean image to Canvas");
					this.wasmLightMapper.AverageSamplesAndRenderToCanvas();
					imageData = this.canvas.toDataURL("image/png");
					
					let link = document.createElement('a');
					link.href = imageData;
					link.download = 'noisy.png';
					document.body.appendChild(link);
					link.click(); 
					link.remove();
				});
			}
		}  

		//
		//if we are building a denoising training data set, generate final clean output map and normal and postion map 
		if(deNoiseTrainMode){
			addFunctionToChain(()=>{
				console.log("Rendering Training clean image to Canvas");
				this.wasmLightMapper.AverageSamplesAndRenderToCanvas();
				imageData = this.canvas.toDataURL("image/png");
				
				let link = document.createElement('a');
				link.href = imageData; 
				link.download = 'clean.png';
				document.body.appendChild(link);
				link.click();
				link.remove();
			});
		}
		
		addFunctionToChain(()=>{ 
			averageFrameRenderTime = performance.now() - averageFrameRenderTime; 

			let frameCount = framesPerRender * renderCount; 
			averageFrameRenderTime /= frameCount; 
			console.log("average frame render time " + averageFrameRenderTime + "ms"); 
			
			console.log("Averaging samples");
			this.wasmLightMapper.AverageSamples(); 
		});

		//
		//run filtering and save output 

		for(let i = 0; i < dillateCount; i++){   
			addFunctionToChain(()=>{
				console.log("Dillating Image");
				this.wasmLightMapper.Dilate();
			});
		}

		for(let i = 0; i < gaussFilterCount; i++){    
			addFunctionToChain(()=>{
				console.log("gauss Image");
				this.wasmLightMapper.RunGaussFilter();
			});
		}

		for(let i = 0; i < edgeAwareFilterCount; i++){    
			addFunctionToChain(()=>{
				console.log("Filter Image");
				this.wasmLightMapper.RunEdgeAwareFilter();
			});
		}

		for(let i = 0; i < medianFilterCount; i++){    
			addFunctionToChain(()=>{
				console.log("median  Image");
				this.wasmLightMapper.RunMedianFilter();
			});
		}

		addFunctionToChain(()=>{
			console.log("Rendering to Canvas");
			this.wasmLightMapper.RenderToCanvas();
		});

		addFunctionToChain(()=>{
			this.wasmLightMapper.Stop(); 
		})
		

		addFunctionToChain(()=>{
			console.log("Creating scene data");

			//Prep for Upload the newly created scene data and lightmap to firebase firestore and spacebuilder.
			
			let vecPBuffer = [];
			let vecNBuffer = [];
			let vecUVBuffer = [];
			let vecLMUVBuffer = [];
			let vecIBuffer = [];
			let vecMBuffer = [];
			let vecTBuffer = [];
			let vecMaterialsBuffer = []; 
			let vecMInfoBuffer = [];

			for(let i = 0; i < vecP.size(); i++){
				vecPBuffer.push(vecP.get(i));
			}
			for(let i = 0; i < vecN.size(); i++){
				vecNBuffer.push(vecN.get(i));
			}
			for(let i = 0; i < vecUV.size(); i++){
				vecUVBuffer.push(vecUV.get(i));
			}
			for(let i = 0; i < vecLMUV.size(); i++){
				vecLMUVBuffer.push(vecLMUV.get(i));
			}
			for(let i = 0; i < vecI.size(); i++){
				vecIBuffer.push(vecI.get(i));
			}
			for(let i = 0; i < vecM.size(); i++){
				vecMBuffer.push(vecM.get(i));
			}
			for(let i = 0; i < vecT.size(); i++){
				vecTBuffer.push(vecT.get(i));
			}

			for(let i = 0; i < vecMaterial.size(); i++){
				vecMaterialsBuffer.push(vecMaterial.get(i));
			}

			for(let i = 0; i < vecMInfo.size(); i++){
				vecMInfoBuffer.push(vecMInfo.get(i));
			}

			let sceneData = {
				positions: vecPBuffer, 
				normals: vecNBuffer,
				uvs: vecUVBuffer,
				lmuvs: vecLMUVBuffer,
				indices: vecIBuffer,
				meshes: vecMBuffer,  
				transforms: vecTBuffer,
				materials: vecMaterialsBuffer,
				info: vecMInfoBuffer,
			};

			//Add colors channel to verts in scene data 
			let colors = []; 
            for(let i = 0; i < sceneData.materials.length/2; i++){

                let materialColor =  sceneData.materials[i * 2 + 0]; 
                
                if(!materialColor) 
                    materialColor = '#FFFFFF'; 

                let vLen = sceneData.meshes[i*4 + 3];
                for(let v = 0; v < vLen; v++){
                    let col = Color3.FromHexString(materialColor); 
                    colors.push(col.r, col.g, col.b, 1.0); 
                }
            }
			sceneData.colors = colors; 

			//clean up vectors 
			deleteSceneVecs(); 


			const t1 = performance.now();
			console.log('Took ' + (t1-t0) + ' ms');

			return sceneData; 
		}); 


		//
		//Group meshes 
		addFunctionToChain((combinedSceneData)=>{

			//interactables ??
			//uid, 
			//need to have as interactable as a group, 

			let groups = new Map(); 
			let grouspsMetaData = new Map(); 

			//
			//Add each mesh in the combined mesh data to a group
			for(let m = 0; m < combinedSceneData.meshes.length / 4; m++){
				let ituid = internalTextureUIDs[m]; 

				let meshmetadata = JSON.parse(combinedSceneData.info[m*4 + 3]);
				if(meshmetadata && meshmetadata.unique)
					ituid = m;

				//
				//Get the group from the groups map, if it doesnt exist create the group and add it to the map
				let group;  
				let groupMeta; 
				if(groups.has(ituid)){
					group = groups.get(ituid);
					groupMeta = grouspsMetaData.get(ituid); 
				}
				else{
					group = {
						positions: [], 
						normals: [],
						uvs: [],
						lmuvs: [], 
						colors: [],
						indices: [],
						meshes: [],  
						transforms: [],
						materials: [],
						info: [], 
					}; 

					groupMeta = {
						vCount: 0, 
						iCount: 0,
					}; 

					groups.set(ituid, group); 
					grouspsMetaData.set(ituid, groupMeta);
				}

				//
				//add the mesh data to the group!
				let iOffset = combinedSceneData.meshes[m*4 + 0]; 
				let iLength = combinedSceneData.meshes[m*4 + 1]; 
				let vOffset = combinedSceneData.meshes[m*4 + 2]; 
				let vLength = combinedSceneData.meshes[m*4 + 3];
				 
				for(let i = vOffset; i < vOffset + vLength; i++){
					group.positions.push(combinedSceneData.positions[i*3 + 0], combinedSceneData.positions[i*3 + 1], combinedSceneData.positions[i*3 + 2]); 
					group.normals.push(combinedSceneData.normals[i*3 + 0], combinedSceneData.normals[i*3 + 1], combinedSceneData.normals[i*3 + 2]); 
					group.uvs.push(combinedSceneData.uvs[i*2 + 0], combinedSceneData.uvs[i*2 + 1]); 
					group.lmuvs.push(combinedSceneData.lmuvs[i*2 + 0], combinedSceneData.lmuvs[i*2 + 1]); 
					group.colors.push(combinedSceneData.colors[i*4 + 0], combinedSceneData.colors[i*4 + 1], combinedSceneData.colors[i*4 + 2], 1.0);
				} 

				for(let i = iOffset; i < iOffset + iLength; i++){
					group.indices.push(combinedSceneData.indices[i] - vOffset + groupMeta.vCount); 
				}
				
				group.meshes.push(groupMeta.iCount);
				group.meshes.push(iLength);
				group.meshes.push(groupMeta.vCount);
				group.meshes.push(vLength);

				for(let i = 0; i < 16; i++){
					group.transforms.push( combinedSceneData.transforms[m*16 + i] ); 
				}

				group.materials.push(combinedSceneData.materials[m*2 + 0]); 
				group.materials.push(combinedSceneData.materials[m*2 + 1]); 

				group.info.push(combinedSceneData.info[m*4 + 0]);
				group.info.push(combinedSceneData.info[m*4 + 1]);
				group.info.push(combinedSceneData.info[m*4 + 2]);
				group.info.push(combinedSceneData.info[m*4 + 3]);

				groupMeta.vCount += vLength; 
				groupMeta.iCount += iLength; 
			}

			return Array.from(groups.values());
		}); 

		
		//upload data
		//Remove any baked data
		let imageData = null; 
		return pomiseChain.then((sceneData)=>{
			
			//clear the cancel 
			this.cancel = false; 
			this.progress = 1.0; 
			
			return new Promise((resolve, reject)=>{ 

				resolve(); 

			})
			.then(()=>{
				imageData = this.canvas.toDataURL("image/jpeg");//convert the rendered lightmap canvas to a jpg image and return it
			})
			.then(()=>{

				//Cleanup
				this.canvas.remove(); 
				this.canvas = null; 
				
				this.wasmLightMapper.delete();  


				return {sceneData, lightMap:imageData}; 
			}); 
		}) 

		.catch(()=>{
			//on cancel etc. need to clean up 
			this.wasmLightMapper.Stop(); 
			deleteSceneVecs(); 
			this.canvas.remove(); 
			this.canvas = null; 
			this.wasmLightMapper.delete();
		}); 

		

	}

	Cancel(){
		this.cancel = true; 
	}
};