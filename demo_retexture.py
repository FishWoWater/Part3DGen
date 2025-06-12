import os
import sys

sys.path.append(os.getcwd())
import argparse
import glob

import trimesh

from partgen.generator import PartAware3DGenerator

parser = argparse.ArgumentParser(description="Retexture a part of the structured 3D mesh")
parser.add_argument("--task-dir", required=True, type=str, help="Directory containing the task files")
parser.add_argument("--tex-bake-mode", choices=['opt', 'fast'], default='opt')
parser.add_argument("--retexture-part-id", default=0, type=int, help="Part ID to retexture, -1 to disable re-texture")
parser.add_argument("--retexture-prompt", default='', type=str, help="Prompt for re-texturing the part")
args = parser.parse_args()


generator = PartAware3DGenerator()

task_dir = args.task_dir 
textured_part_dir = os.path.join(task_dir, "textured_parts")
if not os.path.exists(textured_part_dir):
    print(f"Textured part directory does not exist: {textured_part_dir}")
    sys.exit(1)

textured_part_paths = sorted(glob.glob(os.path.join(textured_part_dir, "*.glb")))
if not textured_part_paths:
    print(f"No textured parts found in {textured_part_dir}")
    sys.exit(1)
    
textured_parts = [trimesh.load(path, force='mesh') for path in textured_part_paths]
if args.retexture_part_id < 0 or args.retexture_part_id >= len(textured_parts):
    print(f"Invalid retexture part ID: {args.retexture_part_id}. Valid range is 0 to {len(textured_parts) - 1}.")
    sys.exit(1)
    
part_to_retexture = textured_parts[args.retexture_part_id]
retextured_mesh, retextured_path = generator.retexture_part(part_to_retexture, args.retexture_prompt, part_id=args.retexture_part_id, task_dir=task_dir)

textured_parts[args.retexture_part_id] = retextured_mesh
# recombine with the other submeshes into a scene
recombined_scene = trimesh.Scene(textured_parts)

model_id = 1 
while os.path.exists(os.path.join(task_dir, f"retextured_mesh_{model_id:02d}.glb")):
    model_id += 1
recombined_scene.export(os.path.join(task_dir, f"retextured_mesh_{model_id:02d}.glb"))

normalised_prompt = args.retexture_prompt.strip().lower().replace(" ", "_").replace(",", "_")
retexture_dir = os.path.join(task_dir, f"retexture_parts_{normalised_prompt}")
os.makedirs(retexture_dir, exist_ok=True)

for part_id, part in enumerate(textured_parts):
    part.export(os.path.join(retexture_dir, f"part_{part_id:02d}.glb"))
