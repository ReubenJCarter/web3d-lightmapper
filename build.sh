#!/bin/bash

./emsdk/emsdk activate latest
source ./emsdk/emsdk_env.sh

#compile c++ project
echo "Compiling C++ project..."

#get all cpp files in src folder 
srcFiles=(\
./src/lightmapperWasm.cpp \
)

#make obj build dir if it doesnt exist
mkdir -p ./wasmdist/obj


#compile to obj files
objList=""
for i in "${srcFiles[@]}"
do
    fileName=$(basename -- "$i") 
    objList="${objList} ./wasmdist/obj/${fileName}.o"

    echo building "${fileName}.o"

    #get all dependancies for this obj file 
    depStr=$(emcc $i -MM -I ./src/thirdparty)

    echo calculated deps

    #split string into array of file names, remove any that are invalid 
    IFS=', ' read -r -a deps <<< "$depStr"
    

    #test the time stamp on each file including cpp against tiimestamp of obj file, if any are newer, rebuild 
    shouldBuild=false
    # for depfile in "${deps[@]}"
    # do
    #     if [ ./${depfile} -nt ./wasmdist/obj/${fileName}.o ]
    #         then
    #             shouldBuild=true
    #             echo $depfile NEWER
    #     fi
    # done
      shouldBuild=true
    
    if [ "$shouldBuild" = true ] 
    then
        echo building...
        if [ "$1" == "prod" ]
        then
            emcc -c -std=c++17 -O3 -o ./wasmdist/obj/$fileName.o -I ./src/thirdparty $i &
        else
            emcc -c -std=c++17 -g -gsource-map -o ./wasmdist/obj/$fileName.o -I ./src/thirdparty $i &
        fi
    fi
done

wait 

#run linking 
echo linking...
if [ "$1" == "prod" ]
then
    emcc -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s ENVIRONMENT='web' -s USE_ES6_IMPORT_META=0 -s EXPORT_ES6=1 -s MAX_WEBGL_VERSION=2 -s --bind -std=c++17 -O3 -o ./wasmdist/lightmapperWasm.mjs -I ./src/thirdparty $objList
else
    emcc -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s ENVIRONMENT='web' -s USE_ES6_IMPORT_META=0 -s EXPORT_ES6=1 -s MAX_WEBGL_VERSION=2 -s GL_ASSERTIONS=1 -s ASSERTIONS=1 --bind -std=c++17 -g -gsource-map -o ./wasmdist/lightmapperWasm.mjs -I ./src/thirdparty $objList
fi

#copy wasm file to test build folder 
echo copy test files...
mkdir -p ./test/build/
cp ./wasmdist/lightmapperWasm.wasm ./test/build/
cp ./wasmdist/lightmapperWasm.wasm.map ./test/build/
cp ./wasmdist/lightmapperWasm.mjs ./test/build/
cp ./src/lightmapperBabylon.js ./wasmdist/
cp ./wasmdist/lightmapperBabylon.js ./test/build/

#copy test models to build folder
mkdir -p ./test/build/models
cp -r ./test/models/* ./test/build/models 


read -p "PRESS ANY KEY..."