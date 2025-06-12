# download feature field 
mkdir -p pretrained/PartField
wget https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt -O pretrained/PartField/model_objaverse.pt

# download hunyuan 
mkdir -p pretrained/tencent/Hunyuan3D-2
huggingface-cli download --resume-download tencent/Hunyuan3D-2 --include hunyuan3d-dit-v2-0-turbo/* hunyuan3d-vae-v2-0-turbo/* --local-dir pretrained/tencent/Hunyuan3D-2

mkdir -p pretrained/tencenet/Hunyuan3D-2mini
huggingface-cli download --resume-download tencent/Hunyuan3D-2mini --include hunyuan3d-dit-v2-mini-turbo/* hunyuan3d-vae-v2-mini-turbo/* --local-dir pretrained/tencent/Hunyuan3D-2mini

# download trellis 
mkdir -p pretrained/TRELLIS/TRELLIS-image-large
huggingface-cli download --resume-download microsoft/TRELLIS-image-large --local-dir pretrained/TRELLIS/TRELLIS-image-large

# optional, if you require text-conditioned part re-texturing
# mkdir -p pretrained/TRELLIS/TRELLIS-text-xlarge
# huggingface-cli download --resume-download microsoft/TRELLIS-text-xlarge --local-dir pretrained/TRELLIS/TRELLIS-text-xlarge

# download holopart 
mkdir -p pretrained/HoloPart
huggingface-cli download --resume-download VAST-AI/HoloPart --local-dir pretrained/HoloPart
