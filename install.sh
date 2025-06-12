cd thirdparty/TRELLIS
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
# for systems with glibc < 2.29 , you may need to build kaolin from source manually

cd ../../thirdparty/PartField 
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d psutil 
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# install hunyuan3d 
cd ../../thirdparty/Hunyuan3D-2
pip install -r requirements.txt 
pip install -e .
# we are only going to use hunyuan turbo for geometry generation
# so no need to install the texture relevant depdendencies

# optional, only required if you are going to use part completion 
cd ../../thirdparty/HoloPart
pip install -r requirements.txt 

# numpy compability
pip install numpy==1.24.4
pip install pydantic==2.10.6
pip install gradio_litmodel3d

cd ../..

# download all models 
./download_models.sh 
# do sanity check 
./sanity_check.sh
