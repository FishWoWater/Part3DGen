import os
import sys

sys.path.append(os.getcwd())
import argparse
import glob

from PIL import Image

from partgen.generator import PartAware3DGenerator

parser = argparse.ArgumentParser(description="Part-aware 3D object generator")
parser.add_argument("--image-path", default='', type=str, help="Path to the model file")
parser.add_argument("--image-dir", default='', type=str, help="Directory containing the image files")
parser.add_argument("--tex-image-path", default='', type=str, help="Path to the texture image file")
parser.add_argument("--output-dir", default='exp_results/pipeline', type=str, help="Root directory to save the output files")
parser.add_argument("--num-max-parts", default=10, type=int, help="Maximum number of parts to segment")
parser.add_argument("--num-parts", default=10, type=int, help="number of parts to segment")
parser.add_argument("--enable-part-completion", action='store_true', help="Enable part completion")
parser.add_argument("--trellis-bake-decoder", choices=['rf', 'gaussian'], default='gaussian', help="Decoder to use for Trellis baking")
parser.add_argument("--num-holopart-steps", default=50, type=int, help="Number of inference steps for Holopart")
parser.add_argument("--num-trellis-steps", default=12, type=int, help="Number of inference steps for Trellis")
parser.add_argument("--simplify", default=0.95, type=float, help="Simplify ratio for mesh decimation")
parser.add_argument("--low-vram", action='store_true', help="Enable low VRAM mode for Trellis")
parser.add_argument("--tex-bake-mode", choices=['opt', 'fast'], default='opt')
parser.add_argument("--use-hunyuan-mini", default=False, action='store_true', help="Use Hunyuan Mini model for part completion")
parser.add_argument("--retexture-part-id", default=-1, type=int, help="Part ID to retexture, -1 to disable re-texture")
parser.add_argument("--retexture-prompt", default='', type=str, help="Prompt for re-texturing the part")
args = parser.parse_args()

assert args.image_path or args.image_dir, "Please provide either --image-path or --image-dir"

generator = PartAware3DGenerator(saveroot=args.output_dir, 
                                 use_hunyuan_mini=args.use_hunyuan_mini, 
                                 enable_part_completion=args.enable_part_completion, 
                                 low_vram=args.low_vram, 
                                 trellis_bake_decoder=args.trellis_bake_decoder,)

if args.image_dir:
    image_paths = glob.glob(os.path.join(args.image_dir, "*.png"))
else:
    image_paths = [args.image_path] 
    
for image_path in image_paths:
    print(f"Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Image file does not exist: {image_path}")
        continue
    tex_image_path = args.tex_image_path or image_path

    # image = Image.open(image_path).convert("RGB")
    # tex_image = Image.open(tex_image_path).convert("RGB")

    tex_mesh, tex_submeshes = generator.run(image_path, tex_image_path, 
                enable_part_completion=args.enable_part_completion,
                num_holopart_steps=args.num_holopart_steps, 
                num_trellis_steps=args.num_trellis_steps, 
                num_max_parts=args.num_max_parts, 
                num_parts=args.num_parts, 
                trellis_bake_mode=args.tex_bake_mode, 
                simplify_ratio=args.simplify)

    if args.retexture_part_id >= 0 and args.retexture_prompt:
        print(f"Retexturing part {args.retexture_part_id} with prompt: {args.retexture_prompt}")
        generator.retexture_part(args.retexture_part_id, args.retexture_prompt)
    else:
        print("Skipping re-texture as part ID is -1 or prompt is empty.")
