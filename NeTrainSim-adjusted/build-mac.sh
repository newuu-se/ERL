cmake -B build-mac -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=OFF -DBUILD_SERVER=OFF -DCMAKE_PREFIX_PATH=$(brew --prefix qt6)
cmake --build build-mac --target NeTrainSimConsole -j$(sysctl -n hw.logicalcpu)
mkdir -p res
./build-mac/src/NeTrainSimConsole/NeTrainSim \
  -n src/data/sampleProject/nodesFile.dat \
  -l src/data/sampleProject/linksFile.dat \
  -t src/data/sampleProject/dieselTrain.dat \
  -o res
