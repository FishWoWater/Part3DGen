#!/usr/bin/env python
# coding=utf-8

import os
import random
import shutil
import time
import uuid

import gradio as gr
import numpy as np
import torch
import trimesh
from gradio_litmodel3d import LitModel3D

from partgen.generator import PartAware3DGenerator
from partgen.utils import voxelize

# Constants
MAX_SEED = 1e7
EXAMPLE_IMAGES_DIR = "assets/example_image"


def randomize_seed(seed: int, randomize_seed: bool) -> int:
    """Randomize seed if requested"""
    if randomize_seed:
        seed = random.randint(0, int(MAX_SEED))
    return seed


class PartAware3DApp:

    def __init__(self):
        self.generator = None
        self.current_hunyuan_mini = False
        self.current_part_completion = True
        self.cached_geo_image_hash = None
        self.cached_results = None
        # Don't initialize generator immediately - delay until first use        # State for part re-texturing
        self.current_task_dir = None
        self.current_textured_mesh = None
        self.current_submesh_segment_indices = None
        self.current_submeshes = None
        
        # State for part viewing
        self.current_raw_part_paths = []
        self.current_completed_part_paths = []
        self.current_textured_part_paths = []

    def _compute_image_hash(self, image):
        """Compute a hash of the image content for comparison."""
        if image is None:
            return None
        small_img = image.convert("L").resize((64, 64))
        return hash(small_img.tobytes())

    def _initialize_generator(self, use_hunyuan_mini=False, enable_part_completion=True):
        # Check if we need to reinitialize
        if (self.generator is not None and use_hunyuan_mini == self.current_hunyuan_mini and
                enable_part_completion == self.current_part_completion):
            return

        # Cleanup old generator if it exists
        if self.generator is not None:
            del self.generator
            torch.cuda.empty_cache()

        # Create new generator with delayed initialization
        self.generator = PartAware3DGenerator(
            use_hunyuan_mini=use_hunyuan_mini,
            save_gs_video=False,
            require_rembg=True,
            # adjust this if you have sufficient VRAM
            low_vram=True,
            enable_part_completion=enable_part_completion,
            device="cuda",
            saveroot="exp_results/pipeline",
            verbose=True,
        )

        # Update current settings
        self.current_hunyuan_mini = use_hunyuan_mini
        self.current_part_completion = enable_part_completion

    def process_image(
        self,
        image,
        image_tex,
        use_hunyuan_mini,
        enable_part_completion,
        num_parts,
        num_max_parts,
        simplify_ratio,
        num_holopart_steps,
        num_trellis_steps,
        trellis_cfg,
        trellis_bake_mode,
        fill_mesh_holes,
        seed,
        randomize_seed_checkbox):
        """Process image through the complete Part-Aware 3D pipeline"""

        if image is None:
            yield None, None, None, None, None, None, None, "Please upload an image first."
            return

        # Initialize generator with current settings
        self._initialize_generator(use_hunyuan_mini, enable_part_completion)

        tik_pipe = time.time()
        torch.cuda.reset_peak_memory_stats()

        # Randomize seed if requested
        actual_seed = randomize_seed(seed, randomize_seed_checkbox)

        # Use geometry image for texture if no texture image provided
        texture_image = image_tex if image_tex is not None else image

        try:            
            # Stage 1: Generate geometry
            yield None, None, None, None, None, None, None, "🔄 Stage 1/4: Generating geometry with Hunyuan3D..."

            # Create task directory structure
            task_uid = str(uuid.uuid4())
            task_dir = os.path.join(self.generator.saveroot, task_uid)
            os.makedirs(task_dir, exist_ok=True)

            # Step 1: Generate raw geometry
            raw_mesh, raw_geometry_path = self.generator.geometry_generation(image,
                                                                             task_dir,
                                                                             simplify_ratio=simplify_ratio)            # Show raw geometry immediately
            scene_binary_voxel = voxelize(raw_mesh, is_input_yup=True, do_normalization=False)

            # do decimation
            decimated_mesh, decimated_path = self.generator.decimate_and_postprocess(
                raw_mesh,
                task_dir=task_dir,
                simplify_ratio=simplify_ratio,
                fill_mesh_holes=True)
            yield decimated_path, None, None, None, None, None, None, "✅ Stage 1/4: Geometry generated! Starting mesh segmentation..."

            # Stage 2: Mesh segmentation
            scene, scene_path, submeshes, _, raw_part_paths, segmented_mesh_path = self.generator.run_mesh_segmentation(
                decimated_mesh,
                decimated_path,
                task_dir=task_dir,
                num_parts=num_parts,
                num_max_parts=num_max_parts)            # Show assembled scene and raw parts immediately
            yield decimated_path, scene_path, segmented_mesh_path, None, raw_part_paths[0] if raw_part_paths else None, None , None, "✅ Stage 2/4: Mesh segmented! Starting part completion..."

            # copy the segmented mesh file to the task directory
            if os.path.exists(segmented_mesh_path):
                shutil.copy(segmented_mesh_path, os.path.join(task_dir, f"segmented_mesh_{num_parts}.obj"))
                print("Copying segmented mesh to task directory:", os.path.join(task_dir, f"segmented_mesh_{num_parts}.obj"))

            # Stage 3: Optional part completion
            if enable_part_completion:
                scene, completed_scene_path, submeshes, completed_part_paths = self.generator.run_part_completion(
                    scene_path, task_dir=task_dir, num_holopart_steps=num_holopart_steps, seed=actual_seed)
                scene_as_mesh = trimesh.util.concatenate(submeshes)                # Show completed scene
                yield decimated_path, completed_scene_path, segmented_mesh_path, None, raw_part_paths[0] if raw_part_paths else None, completed_part_paths[0] if completed_part_paths else None, None, "✅ Stage 3/4: Parts completed! Starting texture generation..."
            else:
                completed_part_paths = raw_part_paths.copy()
                scene_as_mesh = trimesh.util.concatenate(submeshes)
                yield decimated_path, scene_path, segmented_mesh_path, None, raw_part_paths[0] if raw_part_paths else None, completed_part_paths[0] if completed_part_paths else None , None, "✅ Stage 3/4: Skipped part completion! Starting texture generation..."

            # Stage 4: Texture generation
            submesh_segment_indices = np.cumsum([len(submesh.faces) for submesh in submeshes])

            tex_mesh, textured_mesh_path = self.generator.tex_generation(scene_as_mesh,
                                                                         image_tex=texture_image,
                                                                         voxel=scene_binary_voxel,
                                                                         num_trellis_steps=num_trellis_steps,
                                                                         trellis_cfg=trellis_cfg,
                                                                         trellis_bake_mode=trellis_bake_mode,
                                                                        #  fill_mesh_holes=fill_mesh_holes,
                                                                         task_dir=task_dir)            # Split textured mesh into parts
            tex_submeshes = self.generator._split_textured_mesh(tex_mesh, submesh_segment_indices,
                                                                task_dir)  
            
            # Store state for part re-texturing
            self.current_task_dir = task_dir
            self.current_textured_mesh = tex_mesh
            self.current_submesh_segment_indices = submesh_segment_indices
            self.current_submeshes = tex_submeshes

            # Get textured part paths
            textured_parts_dir = os.path.join(task_dir, "textured_parts")
            textured_part_paths = []
            if os.path.exists(textured_parts_dir):
                textured_part_paths = [
                    os.path.join(textured_parts_dir, f)
                    for f in sorted(os.listdir(textured_parts_dir))
                    if f.endswith('.glb')
                ]

            # Store parts for individual viewing
            self.current_raw_part_paths = raw_part_paths
            self.current_completed_part_paths = completed_part_paths
            self.current_textured_part_paths = textured_part_paths

            # Return first parts for initial display (or None if no parts)
            first_raw_part = raw_part_paths[0] if raw_part_paths else None
            first_completed_part = completed_part_paths[0] if completed_part_paths else None
            first_textured_part = textured_part_paths[0] if textured_part_paths else None

            tok = time.time()
            max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024

            meta = (f"✅ Pipeline completed successfully!\n"
                    f"⏱️ Time: {tok - tik_pipe:.1f}s | 🧠 Peak Memory: {max_mem:.1f}GB | 🎲 Seed: {actual_seed}\n"
                    f"📊 Generated {len(tex_submeshes)} parts")            # Final yield with all results
            final_scene_path = completed_scene_path if enable_part_completion and 'completed_scene_path' in locals(
            ) else scene_path
            yield (decimated_path, final_scene_path, segmented_mesh_path, textured_mesh_path, first_raw_part, first_completed_part, first_textured_part, meta)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"❌ Error during processing: {str(e)}"
            yield None, None, None, None, None, None, None, error_msg

    def retexture_selected_part(
        self,
        part_selection: int,
        text_prompt: str,
        num_trellis_steps: int = 12,
        trellis_cfg: float = 7.5,
        trellis_bake_mode: str = "fast",
    ):
        """Re-texture a selected part with the given text prompt."""

        if self.generator is None:
            return None, None, "❌ No model loaded. Please run the main pipeline first."

        if self.current_textured_mesh is None or self.current_submeshes is None:
            return None, None, "❌ No textured mesh available. Please run the main pipeline first."

        if not text_prompt.strip():
            return None, None, "❌ Please enter a text prompt for re-texturing."

        if part_selection < 0 or part_selection >= len(self.current_submeshes):
            return None, None, f"❌ Invalid part selection. Please choose between 0 and {len(self.current_submeshes)-1}."

        try:
            # Get the selected part mesh
            selected_part = self.current_submeshes[part_selection]

            # Re-texture the selected part
            retextured_part, retextured_part_path = self.generator.retexture_part(
                part_mesh=selected_part,
                text_prompt=text_prompt,
                num_trellis_steps=num_trellis_steps,
                trellis_cfg=trellis_cfg,
                trellis_bake_mode=trellis_bake_mode,
                task_dir=self.current_task_dir,
                part_id=part_selection,
            )

            # Combine the re-textured part back into the full mesh
            combined_mesh, combined_mesh_path = self.generator.combine_retextured_part(
                full_textured_mesh=self.current_textured_mesh,
                retextured_part=retextured_part,
                part_id=part_selection,
                submesh_segment_indices=self.current_submesh_segment_indices,
                task_dir=self.current_task_dir,
            )

            success_msg = (f"✅ Part {part_selection} re-textured successfully!\n"
                           f"🎨 Prompt: '{text_prompt}'\n"
                           f"📁 Saved to: {retextured_part_path}")

            return retextured_part_path, combined_mesh_path, success_msg

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"❌ Error during part re-texturing: {str(e)}"
            return None, None, error_msg

    def get_raw_part(self, part_id: int):
        """Get a raw part by ID for display"""
        if not self.current_raw_part_paths or part_id < 0 or part_id >= len(self.current_raw_part_paths):
            return None
        return self.current_raw_part_paths[part_id]

    def get_completed_part(self, part_id: int):
        """Get a completed part by ID for display"""
        if not self.current_completed_part_paths or part_id < 0 or part_id >= len(self.current_completed_part_paths):
            return None
        return self.current_completed_part_paths[part_id]
    
    def get_segmented_mesh_path(self, num_parts: int):
        # return os.path.join(self.current_task_dir, f"segmented_mesh_{num_parts}.obj") if self.current_task_dir else None 
        ply_path = os.path.join(self.current_task_dir, "clustering", "ply", f"decimated_mesh_0_{num_parts:02d}.ply")
        if not os.path.exists(ply_path):
            return None
        trimesh.load(ply_path).export(os.path.splitext(ply_path)[0] + ".obj")
        return os.path.splitext(ply_path)[0] + ".obj"

    def get_textured_part(self, part_id: int):
        """Get a textured part by ID for display"""
        if not self.current_textured_part_paths or part_id < 0 or part_id >= len(self.current_textured_part_paths):
            return None
        return self.current_textured_part_paths[part_id]

    def update_part_selectors(self):
        """Update part selector ranges based on current parts"""
        if not self.current_raw_part_paths:
            return gr.Slider(maximum=0), gr.Slider(maximum=0), gr.Slider(maximum=0)
        
        max_parts = len(self.current_raw_part_paths) - 1
        return (
            gr.Slider(minimum=0, maximum=max_parts, value=0),
            gr.Slider(minimum=0, maximum=max_parts, value=0),
            gr.Slider(minimum=0, maximum=max_parts, value=0)
        )


def create_demo():
    app = PartAware3DApp()

    with gr.Blocks(title="Part-Aware 3D Generation Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 Part-Aware 3D Generation Pipeline
        Generate 3D models with semantic parts from a single image using **Hunyuan3D**, **PartField**, **HoloPart**, and **TRELLIS**.
        
        **Pipeline Overview:**
        1. 🏗️ **Geometry Generation**: Create 3D mesh from input image using Hunyuan3D   
        2. ✂️ **Mesh Segmentation**: Split mesh into semantic parts using PartField  
        3. 🔧 **Part Completion**: Optionally complete/refine parts using HoloPart       
        4. 🎨 **Texture Generation**: Add realistic textures using TRELLIS
        5. 🧩 **Part Editing**: Re-texture individual parts with custom prompts
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input Images")

                with gr.Row():
                    # Geometry input
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            type="pil",
                            label="🏗️ Geometry Input Image",
                            height=250,
                        )

                        # Check if example directory exists before creating examples
                        if os.path.exists("assets/example_image"):
                            example_files = [
                                f for f in sorted(os.listdir("assets/example_image"))[:12]
                                if f.endswith(('.png', '.jpg', '.jpeg'))
                            ]
                            if example_files:
                                geo_examples = gr.Examples(
                                    examples=[f"assets/example_image/{image}" for image in example_files],
                                    inputs=[input_image],
                                    label="Example Images (Geometry)",
                                    examples_per_page=8,
                                )

                    # Texture input (optional)
                    with gr.Column(scale=1):
                        texture_image = gr.Image(
                            type="pil",
                            label="🎨 Texture Input Image (Optional)",
                            height=250,
                        )

                        # Check if example directory exists before creating examples
                        if os.path.exists("assets/example_image"):
                            example_files = [
                                f for f in sorted(os.listdir("assets/example_image"))[:12]
                                if f.endswith(('.png', '.jpg', '.jpeg'))
                            ]
                            if example_files:
                                tex_examples = gr.Examples(
                                    examples=[f"assets/example_image/{image}" for image in example_files],
                                    inputs=[texture_image],
                                    label="Example Images (Texture)",
                                    examples_per_page=8,
                                )

                gr.Markdown("### ⚙️ Pipeline Settings")

                with gr.Accordion("🏗️ Geometry Settings", open=True):
                    with gr.Row():
                        use_hunyuan_mini = gr.Checkbox(value=False,
                                                       label="Use Hunyuan Mini",
                                                       info="Faster but lower quality geometry")
                        simplify_ratio = gr.Slider(minimum=0.5,
                                                   maximum=0.99,
                                                   value=0.95,
                                                   step=0.025,
                                                   label="Mesh Simplification Ratio",
                                                   info="Higher values preserve more details")

                with gr.Accordion("✂️ Segmentation Settings", open=True):
                    with gr.Row():
                        num_parts = gr.Slider(minimum=2,
                                              maximum=15,
                                              value=8,
                                              step=1,
                                              label="Number of Parts",
                                              info="Final number of parts to generate")
                        num_max_parts = gr.Slider(minimum=5,
                                                  maximum=20,
                                                  value=10,
                                                  step=1,
                                                  label="Max Parts (Internal)",
                                                  info="Maximum parts for hierarchical segmentation")

                with gr.Accordion("🔧 Part Completion Settings", open=False):
                    enable_part_completion = gr.Checkbox(value=False,
                                                         label="Enable Part Completion",
                                                         info="Use HoloPart to refine and complete parts")
                    num_holopart_steps = gr.Slider(minimum=10,
                                                   maximum=50,
                                                   value=25,
                                                   step=5,
                                                   label="HoloPart Inference Steps",
                                                   info="More steps = better quality, slower generation")

                with gr.Accordion("🎨 Texture Settings", open=False):
                    with gr.Row():
                        num_trellis_steps = gr.Slider(
                            minimum=5,
                            maximum=25,
                            value=12,
                            step=1,
                            label="TRELLIS Steps",
                        )
                        trellis_cfg = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=3.75,
                            step=0.25,
                            label="Image-Conditioned Guidance Scale",
                        )
        
                    with gr.Row():
                        trellis_bake_mode = gr.Radio(choices=["fast", "opt"],
                                                     value="fast",
                                                     label="Texture Bake Mode",
                                                     info="Fast vs optimized texture baking")
                        fill_mesh_holes = gr.Checkbox(value=True, label="Fill Mesh Holes")

                with gr.Accordion("🎲 Random Settings", open=False):
                    with gr.Row():
                        seed = gr.Number(value=2025, label="Seed", precision=0)
                        randomize_seed_checkbox = gr.Checkbox(value=False, label="Randomize Seed")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Pipeline Status")
                output_info = gr.Textbox(label="Generation Progress",
                                         value="Ready to generate! Upload an image and click the button below.",
                                         interactive=False,
                                         lines=3)

                gr.Markdown("### 🔍 3D Model Visualization")

                with gr.Tabs():
                    with gr.TabItem("🏗️ Raw Geometry"):
                        raw_geometry_model = LitModel3D(
                            label="Raw Geometry (Untextured)",
                            height=350,
                        )                    
                    with gr.TabItem("🔗 Assembled Scene"):
                        assembled_scene_model = LitModel3D(
                            label="Assembled Scene (All Parts)",
                            height=350,
                        )
                        
                    with gr.TabItem("✂️ Segmented Model"):
                        segmented_model = LitModel3D(
                            label="Segmented Model",
                            height=350,
                        )

                    with gr.TabItem("🎨 Final Textured Model"):
                        textured_model = LitModel3D(
                            label="Final Textured Model",
                            height=350,
                        )

                gr.Markdown("### 🧩 Individual Parts")

                with gr.Tabs():
                    with gr.TabItem("🔲 Raw Parts"):
                        gr.Markdown("Navigate through individual raw parts using the slider below.")
                        raw_part_selector = gr.Slider(
                            minimum=0,
                            maximum=7,
                            value=0,
                            step=1,
                            label="Select Part ID",
                            info="Choose which part to view"
                        )
                        raw_part_viewer = LitModel3D(
                            label="Raw Part (Untextured)",
                            height=300,
                        )
                        
                    with gr.TabItem("🔲 Completed Parts"):
                        gr.Markdown("Navigate through individual completed parts using the slider below.")
                        completed_part_selector = gr.Slider(
                            minimum=0,
                            maximum=7,
                            value=0,
                            step=1,
                            label="Select Part ID",
                            info="Choose which completed part to view"
                        )
                        completed_part_viewer = LitModel3D(
                            label="Completed Part (Untextured)",
                            height=300,
                        )

                    with gr.TabItem("🎨 Textured Parts"):
                        gr.Markdown("Navigate through individual textured parts using the slider below.")
                        textured_part_selector = gr.Slider(
                            minimum=0,
                            maximum=7,
                            value=0,
                            step=1,
                            label="Select Part ID",
                            info="Choose which part to view"
                        )
                        textured_part_viewer = LitModel3D(
                            label="Textured Part",
                            height=300,
                        )

        # Main generation button
        with gr.Row():
            with gr.Column(scale=1):
                process_btn = gr.Button("🚀 Generate 3D Model", variant="primary", size="lg")
            with gr.Column(scale=1):
                clear_btn = gr.Button("🗑️ Clear Results", variant="secondary", size="lg")

        # Part Re-texturing Section
        gr.Markdown("---")
        gr.Markdown("### 🎨 Part-Specific Re-texturing")
        gr.Markdown("Select a specific part and enter a text prompt to re-texture only that part.")

        with gr.Row():
            with gr.Column(scale=1):
                part_selection = gr.Number(
                    value=0,
                    label="Part ID to Re-texture",
                    info="Enter the ID of the part you want to re-texture (0-based index)",
                    precision=0,
                    minimum=0,
                )
                part_text_prompt = gr.Textbox(
                    label="Text Prompt for Part",
                    placeholder="e.g., 'wooden texture', 'metallic surface', 'red paint'...",
                    lines=2,
                )

                with gr.Accordion("🎛️ Part Re-texturing Settings", open=False):
                    part_num_trellis_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=12,
                        step=1,
                        label="TRELLIS Steps (Part)",
                    )
                    part_trellis_cfg = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=7.5,
                        step=0.25,
                        label="Guidance Scale (Part)",
                    )
                    part_trellis_bake_mode = gr.Radio(
                        choices=["fast", "opt"],
                        value="fast",
                        label="Texture Bake Mode (Part)",
                    )

            with gr.Column(scale=1):
                retexture_info = gr.Textbox(
                    label="Re-texturing Status",
                    value="Ready for part re-texturing. Run the main pipeline first to enable this feature.",
                    interactive=False,
                    lines=3,
                )

                with gr.Tabs():
                    with gr.TabItem("🎨 Re-textured Part"):
                        retextured_part_model = LitModel3D(
                            label="Re-textured Individual Part",
                            height=300,
                        )
                    with gr.TabItem("🔗 Updated Full Model"):
                        updated_full_model = LitModel3D(
                            label="Full Model with Re-textured Part",
                            height=300,
                        )

        with gr.Row():
            retexture_btn = gr.Button("🎨 Re-texture Selected Part", variant="primary", size="lg")        # Set up processing
        process_result = process_btn.click(
            fn=app.process_image,
            inputs=[
                input_image,
                texture_image,
                use_hunyuan_mini,
                enable_part_completion,
                num_parts,
                num_max_parts,
                simplify_ratio,
                num_holopart_steps,
                num_trellis_steps,
                trellis_cfg,
                trellis_bake_mode,
                fill_mesh_holes,
                seed,
                randomize_seed_checkbox,
            ],
            outputs=[
                raw_geometry_model,
                assembled_scene_model,
                segmented_model,
                textured_model,
                raw_part_viewer,
                completed_part_viewer,
                textured_part_viewer,
                output_info,
            ],
        )
        
        # Update part selectors after processing completes
        process_result.then(
            fn=app.update_part_selectors,
            inputs=[],
            outputs=[raw_part_selector, completed_part_selector, textured_part_selector]
        )

        # Set up part re-texturing
        retexture_btn.click(
            fn=app.retexture_selected_part,
            inputs=[
                part_selection,
                part_text_prompt,
                part_num_trellis_steps,
                part_trellis_cfg,
                part_trellis_bake_mode,
            ],
            outputs=[
                retextured_part_model,
                updated_full_model,
                retexture_info,
            ],
        )        # Clear function
        def clear_all():
            return (None, None, None, None, None, None, "Results cleared. Ready for new generation!")

        clear_btn.click(fn=clear_all,
                        outputs=[
                            raw_geometry_model,
                            assembled_scene_model,
                            segmented_model,
                            textured_model,
                            raw_part_viewer,
                            textured_part_viewer,
                            output_info,
                        ])        # Auto-update max parts constraint
        
        # def update_num_parts(num_parts_val):
            # we needs to change the preview segmented path 
            # return gr.Slider(minimum=1, value=num_parts_val)
    
        # def update_max_parts(num_parts_val):
            # return gr.Slider(minimum=max(5, num_parts_val), value=max(10, num_parts_val))

        # num_parts.change(fn=update_num_parts, inputs=[num_parts], outputs=[num_parts])
        # TODO: finish this
        num_parts.change(fn=app.get_segmented_mesh_path, inputs=[num_parts], outputs=[segmented_model])
        # FIX this
        # num_max_parts.change(fn=update_max_parts, inputs=[num_max_parts], outputs=[num_max_parts])

        # Wire up part selectors
        raw_part_selector.change(
            fn=app.get_raw_part,
            inputs=[raw_part_selector],
            outputs=[raw_part_viewer]
        )

        completed_part_selector.change(
            fn=app.get_completed_part,
            inputs=[completed_part_selector],
            outputs=[completed_part_viewer]
        )
        
        textured_part_selector.change(
            fn=app.get_textured_part,
            inputs=[textured_part_selector],
            outputs=[textured_part_viewer]
        )


    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, show_error=True)
