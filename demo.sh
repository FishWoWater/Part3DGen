# debug mode, only to check that the pipeline can run 
# it has the minimal memory requirement
python demo.py --image-path assets/example_image/typical_humanoid_mech.png \
    --use-hunyuan-mini \
    --tex-bake-mode fast \
    --low-vram\
    --num-max-parts 5


# normal mode, save memory but result is still good 
python demo.py --image-path assets/example_image/typical_humanoid_mech.png \
     --simplify 0.95\
     --tex-bake-mode opt \
     --num-max-parts 6 \
     --num-parts 6\
     --low-vram

# 24GB+ VRAM 
# normal mode, save memory but result is still good 
python demo.py --image-path assets/example_image/typical_humanoid_mech.png \
     --simplify 0.95\
     --tex-bake-mode opt \
     --enable-part-completion \
     --num-holopart-steps 50


# normal mode, run for a directory 
python demo.py --image-dir assets/example_image \
     --simplify 0.95\
     --tex-bake-mode opt \
     --num-max-parts 6 \
     --num-parts 6\
     --low-vram \
     --output-dir exp_results/pipeline/wo_part_completion

python demo.py --image-dir assets/example_image \
     --simplify 0.95\
     --tex-bake-mode opt \
     --num-max-parts 6 \
     --num-parts 6\
     --low-vram \
     --enable-part-completion \
     --num-holopart-steps 50 \
     --output-dir exp_results/pipeline/w_part_completion
