import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
from partgen.holopart_runner import HoloPartRunner

parser = argparse.ArgumentParser()
parser.add_argument("--mesh-input", type=str, required=True)
parser.add_argument("--output-dir", type=str, default="./exp_results/holopart/")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-inference-steps", type=int, default=50)
parser.add_argument("--guidance-scale", type=float, default=3.5)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
runner = HoloPartRunner(device="cuda", batch_size=args.batch_size)
runner.run_holopart(
    mesh_input=args.mesh_input,
    seed=args.seed,
    num_inference_steps=args.num_inference_steps,
    guidance_scale=args.guidance_scale).export(os.path.join(args.output_dir, "output.glb"))
