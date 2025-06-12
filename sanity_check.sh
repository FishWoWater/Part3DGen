export PYTHONPATH=$PYTHONPATH:$(pwd)/thirdparty/TRELLIS
export PYTHONPATH=$PYTHONPATH:$(pwd)/thirdparty/PartField
export PYTHONPATH=$PYTHONPATH:$(pwd)/thirdparty/HoloPart
export PYTHONPATH=$PYTHONPATH:$(pwd)/thirdparty/Hunyuan3D-2

python -c "from partfield.model_trainer_pvcnn_only_demo import Model" && echo "PartField dependency installed correctly."
python -c "from trellis.pipelines import TrellisImageTo3DPipeline" && echo "Trellis dependency installed successfully."
python -c "import hy3dgen" && echo "Hunyuan3D library installed successfully."
python -c "from holopart.pipelines.pipeline_holopart import HoloPartPipeline" && echo "HoloPart pipeline installed successfully."
