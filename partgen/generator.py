import json
import logging
import os
import os.path as osp
import shutil
import sys
import time
import uuid

sys.path.insert(0, osp.join(os.getcwd(), "TRELLIS"))
os.environ["ATTN_BACKEND"] = "flash-attn"
# setup the cache directory of hunyuan3d models
os.environ["HY3DGEN_MODELS"] = os.path.abspath(os.path.join(os.getcwd(), "pretrained"))

from typing import List, Literal, Optional, Union

import imageio
import numpy as np
import torch
import trimesh
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils

from .holopart_runner import HoloPartRunner
from .partfield_runner import PartFieldRunner
from .utils import (
    denormalise_mesh,
    normalise_mesh,
    rotate_mesh_to_zup,
    timer_decorator,
    voxelize,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PartAware3DGenerator:

    def __init__(
        self,
        use_hunyuan_mini: bool = True,
        save_gs_video: bool = True,
        require_rembg: bool = True,
        enable_part_completion: bool = False,
        trellis_bake_decoder: Literal['rf', 'gaussian'] = 'gaussian',
        device: str = "cuda",
        saveroot: str = "exp_results/pipeline",
        verbose: bool = True,
        low_vram: bool = True,
    ):
        # Store configuration without initializing heavy models
        self.use_hunyuan_mini = use_hunyuan_mini
        self.save_gs_video = save_gs_video
        self.require_rembg = require_rembg
        self.enable_part_completion = enable_part_completion
        self.trellis_bake_decoder = trellis_bake_decoder
        self.device = device
        self.saveroot = saveroot
        self.verbose = verbose
        self.low_vram = low_vram

        self.trellis_skip_models = [
            "sparse_structure_decoder", "sparse_structure_flow_model",  "slat_decoder_mesh"
        ]
        self.trellis_skip_models.append("slat_decoder_rf" if trellis_bake_decoder == "gaussian" else "slat_decoder_gaussian")
        

        # Initialize model placeholders to None - will be loaded on demand
        self.geometry_pipeline = None
        self.image_tex_pipeline = None
        self.text_tex_pipeline = None
        self.rembg = None

        # Lightweight components can be initialized immediately
        self.holopart_runner = None
        self.partfield_runner = None

        # Track initialization status
        self._geometry_initialized = False
        self._text_to_texture_initialized = False
        self._image_to_texture_initialized = False
        self._rembg_initialized = False
        self._partfield_initialized = False
        self._holopart_initialized = False

        logger.info(f"PartAware3DGenerator created with delayed initialization (device={device})")

    def _ensure_geometry_pipeline(self):
        """Lazy initialization of geometry generation pipeline"""
        if not self._geometry_initialized:
            logger.info("🔄 Initializing Hunyuan3D geometry pipeline...")
            tik = time.time()

            self.geometry_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                ("tencent/Hunyuan3D-2" if not self.use_hunyuan_mini else "tencent/Hunyuan3D-2mini"),
                cache_dir=("./pretrained/Hunyuan3D-2" if not self.use_hunyuan_mini else "./pretrained/Hunyuan3D-2mini"),
                subfolder=("hunyuan3d-dit-v2-0-turbo" if not self.use_hunyuan_mini else "hunyuan3d-dit-v2-mini-turbo"),
                use_safetensors=False,
                device=self.device,
            )
            self.geometry_pipeline.enable_flashvdm(topk_mode="merge")
            self._geometry_initialized = True

            tok = time.time()
            logger.info(f"✅ Hunyuan3D geometry pipeline initialized in {tok - tik:.1f}s")
    
    def _offload_geometry_pipeline(self):
        self._geometry_initialized = False 
        del self.geometry_pipeline
        torch.cuda.empty_cache()

    def _ensure_text_to_texture_pipeline(self):
        """Lazy initialization of text-based texture generation pipeline"""
        if not self._text_to_texture_initialized:
            logger.info("🔄 Initializing TRELLIS text-based texture pipeline...")
            tik = time.time()

            self.text_tex_pipeline = TrellisTextTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS-text-xlarge",
                cache_dir="./pretrained/TRELLIS/TRELLIS-text-xlarge",
                skip_models=self.trellis_skip_models)
            self.text_tex_pipeline.to(self.device)
            self._text_to_texture_initialized = True

            tok = time.time()
            logger.info(f"✅ TRELLIS text-to-texture pipeline initialized in {tok - tik:.1f}s")

    def _ensure_image_to_texture_pipeline(self):
        """Lazy initialization of image-based texture generation pipeline"""
        if not self._image_to_texture_initialized:
            logger.info("🔄 Initializing TRELLIS image-based texture pipeline...")
            tik = time.time()

            self.image_tex_pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS-image-large",
                cache_dir="./pretrained/TRELLIS/TRELLIS-image-large",
                skip_models=self.trellis_skip_models)
            self.image_tex_pipeline.to(self.device)
            self._image_to_texture_initialized = True

            tok = time.time()
            logger.info(f"✅ TRELLIS image-to-texture pipeline initialized in {tok - tik:.1f}s")

    def _ensure_rembg(self):
        """Lazy initialization of background removal"""
        if not self._rembg_initialized and self.require_rembg:
            logger.info("🔄 Initializing background removal...")
            tik = time.time()

            self.rembg = BackgroundRemover()
            self._rembg_initialized = True

            tok = time.time()
            logger.info(f"✅ Background removal initialized in {tok - tik:.1f}s")

    def _ensure_partfield_runner(self):
        """Lazy initialization of PartField runner"""
        if not self._partfield_initialized:
            logger.info("🔄 Initializing PartField runner...")
            tik = time.time()

            self.partfield_runner = PartFieldRunner()
            self._partfield_initialized = True

            tok = time.time()
            logger.info(f"✅ PartField runner initialized in {tok - tik:.1f}s")

    def _ensure_holopart_runner(self):
        """Lazy initialization of HoloPart runner"""
        if not self._holopart_initialized and self.enable_part_completion:
            logger.info("🔄 Initializing HoloPart runner...")
            tik = time.time()

            self.holopart_runner = HoloPartRunner()
            self._holopart_initialized = True

            tok = time.time()
            logger.info(f"✅ HoloPart runner initialized in {tok - tik:.1f}s")

    def _initialize_models(self):
        """Legacy method - now using lazy initialization"""
        # This method is kept for backward compatibility but does nothing
        # Models are now initialized on-demand
        pass

    @timer_decorator
    def geometry_generation(self, image: Union[str, Image.Image], savedir: str, simplify_ratio: float = 0.9):
        # Ensure required models are initialized
        self._ensure_geometry_pipeline()
        self._ensure_rembg()
        
        if isinstance(image, str):
            image = Image.open(image)

        if image.mode == "RGB" and self.require_rembg:
            tik = time.time()
            image = self.rembg(image)
            if savedir:
                image.save(osp.join(savedir, "image_rembg.png"))
            logger.info("rembg finished in {:.3f}s".format(time.time() - tik))

        tik = time.time()
        mesh = self.geometry_pipeline(
            image=image,
            num_inference_steps=5,
            octree_resolution=256,
            num_chunks=80000,
            generator=torch.manual_seed(12345),
            output_type="trimesh",
        )[0]
        # fix some problems and do decimation on the mesh
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()

        material = trimesh.visual.material.PBRMaterial(
            roughnessFactor=1.0,
            baseColorFactor=np.array([127, 127, 127, 255], dtype=np.uint8),
        )
        mesh = trimesh.Trimesh(
            # TODO: understand this rotation op
            vertices=mesh.vertices @ np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            faces=mesh.faces,
            process=False,
            material=material,
        )
        # do normalization to get ready for later trellis voxelization
        mesh, _, _ = normalise_mesh(mesh)
        logger.info("geometry generation finished in {:.3f}s: {}vertices, {}faces".format(
            time.time() - tik, len(mesh.vertices), len(mesh.faces)))
        tik = time.time()

        # save before we rotate as gradio expects a y-up mesh
        savepath = osp.join(savedir, "raw_geometry.glb")
        mesh.export(savepath)

        if self.low_vram:
            logger.info("Offloading geometry pipeline to save VRAM...")
            self._offload_geometry_pipeline()
        return mesh, savepath

    @timer_decorator
    def tex_generation(self,
                       mesh: trimesh.Trimesh,
                       text="",
                       image_tex: Optional[Union[Image.Image, str]] = None,
                       voxel: Optional[np.ndarray] = None,
                       num_trellis_steps: int = 12,
                       trellis_cfg: float = 3.0,
                       trellis_bake_mode: Literal["fast", "opt"] = "opt",
                       get_srgb_texture: bool = False,
                       task_dir: str = ""):

        """
        The input is Y-up models, and the output should also be a Y-up models
        """
        # rotate to z-up for compatibility with trellis
        mesh = mesh.copy()                     
        mesh = rotate_mesh_to_zup(mesh)

        logger.info("Before tex generation, mesh vertices bound: {}/{}".format(mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)))

        tik = time.time()
        if text:
            self._ensure_text_to_texture_pipeline()
            assert self.text_tex_pipeline is not None, "Text-to-3D texture pipeline is not initialized."
            outputs = self.text_to_3d_tex(
                mesh,
                text=text,
                binary_voxel=voxel,
                num_trellis_steps=num_trellis_steps,
                trellis_cfg=trellis_cfg,
            )
        elif image_tex is not None:
            # Ensure texture pipeline and rembg are initialized
            self._ensure_image_to_texture_pipeline()
            self._ensure_rembg()
            if isinstance(image_tex, str):
                image_tex = Image.open(image_tex)
            
            assert self.image_tex_pipeline is not None, "Image-to-3D texture pipeline is not initialized."
            assert isinstance(image_tex, Image.Image), "image_tex should be a PIL Image."
            if image_tex.mode == "RGB" and self.require_rembg:
                image_tex = self.rembg(image_tex)
                tik = time.time()
                if task_dir:
                    # TODO: consider the tex id when exporting the background removed image
                    image_tex.save(osp.join(task_dir, "tex_rembg.png"))
                logger.info("rembg finished in {:.3f}s".format(time.time() - tik))
            outputs = self.image_to_3d_tex(
                mesh,
                image_tex=image_tex,
                binary_voxel=voxel,
                num_trellis_steps=num_trellis_steps,
                trellis_cfg=trellis_cfg,
            )
        logger.info("[TRELLIS-MESH] texture generation finished in {:.3f}s".format(time.time() - tik))

        if self.save_gs_video and task_dir:
            torch.cuda.empty_cache()
            # Render the outputs
            video = render_utils.render_video(outputs[self.trellis_bake_decoder][0])["color"]
            imageio.mimsave(os.path.join(task_dir, f"{self.trellis_bake_decoder}.mp4"), video, fps=30)

        tik = time.time()
        # GLB files can be extracted from the outputs
        # In this to_trimesh, it converts the Z-up mesh to Y-up mesh  
        textured_mesh = postprocessing_utils.to_trimesh(
            outputs[self.trellis_bake_decoder][0],
            mesh,
            vertices=mesh.vertices.astype(np.float32),
            faces=mesh.faces.astype(np.int32),
            # baking parameters
            texture_size=1024,  # Size of the texture used for the GLB
            texture_bake_mode=trellis_bake_mode,
            get_srgb_texture=get_srgb_texture,
            debug=False,
            verbose=self.verbose,
            forward_rot=False)

        logger.info("postprocessing finished in {:.3f}s".format(time.time() - tik))

        savepath = ""
        if task_dir:
            savepath = osp.join(task_dir, "textured_mesh.glb")
            textured_mesh.export(savepath)
            if self.verbose:
                textured_mesh.export(osp.join(task_dir, "textured_mesh.obj"))

        return textured_mesh, savepath

    @timer_decorator
    def text_to_3d_tex(
        self,
        mesh: trimesh.Trimesh,
        text: str,
        voxel: Optional[np.ndarray] = None,
        num_trellis_steps: int = 12,
        trellis_cfg: float = 7.5,
    ):
        # When the voxel is not provided, its generation is left inside `run_variant`
        # Run the pipeline
        outputs = self.text_tex_pipeline.run_variant(
            mesh,
            text,
            binary_voxel=voxel,
            seed=1,
            # more steps, larger cfg
            slat_sampler_params={
                "steps": num_trellis_steps,
                "cfg_strength": trellis_cfg,
            },
            formats=[self.trellis_bake_decoder],
        )
        return outputs

    def image_to_3d_tex(
        self,
        mesh,
        image_tex,
        binary_voxel: Optional[np.ndarray] = None,
        num_trellis_steps: int = 12,
        trellis_cfg: float = 3,
    ):
        if binary_voxel is None:
            # If no binary voxel is provided, voxelize the mesh
            binary_voxel = voxelize(mesh)

        # Run the pipeline
        outputs = self.image_tex_pipeline.run_detail_variation(
            binary_voxel,
            image_tex,
            seed=1,
            # more steps, larger cfg
            slat_sampler_params={
                "steps": num_trellis_steps,
                "cfg_strength": trellis_cfg,
            },
            formats=[self.trellis_bake_decoder],
        )
        return outputs

    def _get_meta_file(self, task_dir: str):
        return os.path.join(task_dir, "meta.json")
    
    def get_segmented_mesh_file(self, task_dir: str, num_parts: int):
        return os.path.join(task_dir, "clustering", "ply", f"decimated_mesh_0_{num_parts:02d}.ply")

    def _check_partfield_already_done(self, task_dir: str, num_parts: int):
        meta_file = self._get_meta_file(task_dir)
        if not os.path.exists(meta_file):
            return False 
        return json.load(open(meta_file, "r")).get("partfield", {}).get("num_max_parts", -1) >= num_parts

    @timer_decorator
    def run_mesh_segmentation(self,
                              raw_mesh: trimesh.Trimesh,
                              mesh_path: str,
                              task_dir: str = "",
                              num_parts: int = 8,
                              num_max_parts: int = 10):
        # TODO: after mesh segmentation, remove the small isolated components
        # TODO: also export the segmented previews, so that the user can preview them at the gradio side
        # Final segmented preview for this number of parts
        segmented_mesh_file = self.get_segmented_mesh_file(task_dir, num_parts)
        # First we check whether the partfield has already been run for this task
        if not self._check_partfield_already_done(task_dir, num_max_parts) or not os.path.exists(segmented_mesh_file):
            # Ensure PartField runner is initialized
            self._ensure_partfield_runner()

            # run PartField to get a face-level segmentation labels
            # a list of length num_max_parts, each element is a list of face levels #NumParts X #NumFaces
            hierarchical_labels = self.partfield_runner.run_partfield(mesh_path,
                                                                    os.path.join(task_dir, "partfield_features"),
                                                                    os.path.join(task_dir, "clustering"),
                                                                    num_max_clusters=num_max_parts, 
                                                                    export_mesh=True)
            np.save(os.path.join(task_dir, "clustering", "hierarchical_labels.npy"), hierarchical_labels)
            # when finished we dump the meta file 
            updated_meta = json.load(open(self._get_meta_file(task_dir), "r")) if os.path.exists(self._get_meta_file(task_dir)) else {}
            updated_meta["partfield"] = {
                "num_max_parts": num_max_parts
            }
            json.dump(updated_meta, open(self._get_meta_file(task_dir), "w"), indent=2)
        else:
            # otherwise we directly load the hierarchical labels
            hierarchical_labels = np.load(os.path.join(task_dir, "clustering", "hierarchical_labels.npy"))
            
        face_labels = hierarchical_labels[num_max_parts -
                                          num_parts] if num_parts <= num_max_parts else hierarchical_labels[-1]
        ulabels = np.unique(face_labels)
        # generate a number of submeshes based on the face labels
        submeshes, submesh_names, submesh_paths = [], [], []
        part_export_dir = os.path.join(task_dir, "raw_parts", str(num_parts))
        os.makedirs(part_export_dir, exist_ok=True)

        face_labels_by_id = [np.nonzero(face_labels == ulabel)[0].tolist() for i, ulabel in enumerate(ulabels)]
        face_labels_by_id = [part for part in face_labels_by_id if len(part) > 0]
        submeshes = raw_mesh.submesh(face_labels_by_id, append=False)
        for i, submesh in enumerate(submeshes):
            submesh_path = os.path.join(part_export_dir, f"part_{i:02d}.glb")
            submesh.export(submesh_path)
            submesh_paths.append(submesh_path)

        # make all submeshes into a scene
        scene = trimesh.Scene()
        for submesh_id, submesh in enumerate(submeshes):
            submesh_name = f"geometry_{submesh_id}"
            scene.add_geometry(submesh, geom_name=submesh_name)
            submesh_names.append(submesh_name)
        scene_path = os.path.join(task_dir, "raw_scene.glb")
        scene.export(scene_path)
        logger.info("Found {} submeshes from mesh segmentation.".format(len(submeshes)))
        
        segmented_mesh = trimesh.load(segmented_mesh_file, force="mesh")
        segmented_mesh.export(os.path.splitext(segmented_mesh_file)[0] + ".obj")
        return scene, scene_path, submeshes, submesh_names, submesh_paths, os.path.splitext(segmented_mesh_file)[0] + ".obj"

    @timer_decorator
    def run_part_completion(self, scene_path: str, task_dir: str = "", num_holopart_steps: int = 35, submesh_names: Optional[List] = None, num_parts: int = 8, seed=2025):
        # Ensure HoloPart runner is initialized
        self._ensure_holopart_runner()

        scene = self.holopart_runner.run_holopart(scene_path, seed=seed, num_inference_steps=num_holopart_steps)
        # re-export the completed one
        scene_path = os.path.join(task_dir, "completed_scene.glb")
        scene.export(scene_path)
        submeshes = list(scene.geometry.values()) if submesh_names is None else [scene.geometry[name] for name in submesh_names]
        submesh_paths = []
        # export the submesh one by one 
        part_export_dir = os.path.join(task_dir, "completed_parts", str(num_parts))
        os.makedirs(part_export_dir, exist_ok=True)
        for i, submesh in enumerate(submeshes):
            submesh_path = os.path.join(part_export_dir, f"part_{i:02d}.glb")
            submesh.export(submesh_path)
            submesh_paths.append(submesh_path)
        return scene, scene_path, submeshes, submesh_paths

    def _split_textured_mesh(self, tex_mesh: trimesh.Trimesh, submesh_segment_indices: List[int], task_dir: str):
        submeshes = []
        start = 0
        submesh_face_sequences = []
        for index in submesh_segment_indices:
            submesh_face_sequences.append(np.arange(start, index))
            start = index
        # now we have the submesh face sequences, we can split the mesh
        submeshes = tex_mesh.submesh(submesh_face_sequences, append=False)

        # export the submeshes
        part_export_dir = os.path.join(task_dir, "textured_parts")
        os.makedirs(part_export_dir, exist_ok=True)
        submesh_paths = []
        for i, submesh in enumerate(submeshes):
            submesh_path = os.path.join(part_export_dir, f"part_{i:02d}.glb")
            submesh.export(submesh_path)
            if self.verbose:
                submesh.export(os.path.join(part_export_dir, f"part_{i:02d}.obj"))
            submesh_paths.append(submesh_path)
        # we also export them as separate geometries 
        trimesh.Scene(submeshes).export(os.path.join(task_dir, "textured_scene.glb"))
        return submeshes
    
    def decimate_and_postprocess(self, raw_mesh: trimesh.Trimesh, task_dir: str, simplify_ratio: float = 0.95, fill_mesh_holes: bool = True):
        decimated_vertices, decimated_faces = postprocessing_utils.postprocess_mesh(
            raw_mesh.vertices,
            raw_mesh.faces,
            postprocess_mode='simplify',
            simplify_ratio=simplify_ratio,
            fill_holes=fill_mesh_holes,
            fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - simplify_ratio)),
            fill_holes_resolution=1024,
            fill_holes_num_views=1000,
            debug=False, verbose=False
        )
        decimated_mesh = trimesh.Trimesh(
            vertices=decimated_vertices,
            faces=decimated_faces)
        decimated_path = os.path.join(task_dir, "decimated_mesh.glb")
        decimated_mesh.export(decimated_path)
        return decimated_mesh, decimated_path

    @timer_decorator
    def run(
        self, 
        image: Union[Image.Image, str],
        image_tex: Union[Image.Image, str],
        # for mesh segmentation
        num_max_parts: int = 10,
        # final # of parts to be used, can be adjusted in gradio
        num_parts: int = 8,
        # for geometry simplification
        simplify_ratio: float = 0.95,
        # for part completion
        num_holopart_steps: int = 35,
        enable_part_completion: bool = False,
        # for texture generation parameter
        num_trellis_steps: int = 12,
        trellis_cfg: float = 3.0,
        trellis_bake_mode: Literal["fast", "opt"] = "opt",
        fill_mesh_holes: bool = True,
        seed=2025,
    ):
        task_uid = str(uuid.uuid4())
        task_dir = os.path.join(self.saveroot, task_uid if isinstance(image, Image.Image) else os.path.splitext(os.path.basename(image))[0])
        os.makedirs(task_dir, exist_ok=True)

        # Step1: run Hunyuan3D to generate the raw mesh (untextured, geometry-only)
        # The generated geometry will be stored at exp_results/pipeline/<task_uid>/raw_geometry.glb
        # It's currently Y-UP and the un-decimated version 
        raw_mesh, rawpath = self.geometry_generation(image, task_dir, simplify_ratio=simplify_ratio)
        scene_binary_voxel = voxelize(raw_mesh, is_input_yup=True, do_normalization=False)

        # Step2: do decimation
        # simple decimation 
        # decimated_mesh = decimate(raw_mesh, simplify_ratio=simplify_ratio, verbose=self.verbose)
        # mesh postprocess with the trellis implentation
        decimated_mesh, decimated_path = self.decimate_and_postprocess(raw_mesh, task_dir, simplify_ratio=simplify_ratio, fill_mesh_holes=fill_mesh_holes)

        # Step3: Do mesh segmentation using PartField
        _, scene_path, submeshes, submesh_names, _, segmented_mesh_file = self.run_mesh_segmentation(decimated_mesh,
                                                                                 decimated_path,
                                                                                 task_dir=task_dir,
                                                                                 num_parts=num_parts,
                                                                                 num_max_parts=num_max_parts)
        # copy the segmented mesh file to the task directory
        if os.path.exists(segmented_mesh_file):
            shutil.copy(segmented_mesh_file, os.path.join(task_dir, f"segmented_mesh_{num_parts}.obj"))

        # Step4(Optional) Use HoloPart to complete the mesh
        if enable_part_completion:
            scene, scene_path, submeshes, _ = self.run_part_completion(scene_path,
                                                                    task_dir=task_dir,
                                                                    num_holopart_steps=num_holopart_steps,
                                                                    submesh_names=submesh_names,
                                                                    seed=seed)
            scene_as_mesh = trimesh.util.concatenate(submeshes)
        else:
            scene_as_mesh = trimesh.util.concatenate(submeshes)

        # Get ready for future segmentation
        submesh_segment_indices = np.cumsum([len(submesh.faces) for submesh in submeshes])

        # Step5: run TRELLIS to texture the mesh
        # Here we texture the mesh as a whole, and optionally TODO we can re-texture each part via image/prompts
        tex_mesh, texpath = self.tex_generation(scene_as_mesh,
                                                # use the voxels from undecimated version
                                                voxel=scene_binary_voxel,
                                                image_tex=image_tex,
                                                num_trellis_steps=num_trellis_steps,
                                                trellis_cfg=trellis_cfg,
                                                trellis_bake_mode=trellis_bake_mode,
                                                task_dir=task_dir)
        logger.info("Final textured mesh saved at: {}".format(texpath))
        # now re-split the textured mesh based on previously recorded ids
        tex_submeshes = self._split_textured_mesh(tex_mesh, submesh_segment_indices, task_dir=task_dir)
        return tex_mesh, tex_submeshes

    @timer_decorator
    def retexture_part(
        self,
        part_mesh: trimesh.Trimesh,
        text_prompt: str,
        num_trellis_steps: int = 12,
        trellis_cfg: float = 7.5,
        trellis_bake_mode: Literal["fast", "opt"] = "fast",
        get_srgb_texture: bool = False,
        task_dir: str = "",
        part_id: int = 0,
    ):
        """
        Re-texture a specific part using text prompt.
        
        Args:
            part_mesh: The mesh of the specific part to re-texture
            text_prompt: Text description for the desired texture
            num_trellis_steps: Number of TRELLIS inference steps
            trellis_cfg: TRELLIS guidance scale
            trellis_bake_mode: Texture baking mode ("fast" or "opt")
            get_srgb_texture: Whether to get sRGB texture
            task_dir: Directory to save results
            part_id: ID of the part being re-textured
        
        Returns:
            tuple: (textured_mesh, save_path)
        """
        if not text_prompt.strip():
            raise ValueError("Text prompt cannot be empty for part re-texturing")
        
        # Ensure text-to-texture pipeline is initialized
        self._ensure_text_to_texture_pipeline()
        assert self.text_tex_pipeline is not None, "Text-to-3D texture pipeline is not initialized."
        
        tik = time.time()
        logger.info(f"🎨 Re-texturing part {part_id} with prompt: '{text_prompt}'")

        part_mesh = part_mesh.copy()
        part_mesh, center, scale = normalise_mesh(part_mesh)
        # rotate to z-up for compatibility with trellis
        part_mesh = rotate_mesh_to_zup(part_mesh)
        voxels = voxelize(part_mesh, is_input_yup=False, do_normalization=False)
        
        # Generate texture using text prompt
        outputs = self.text_to_3d_tex(
            part_mesh,
            text=text_prompt,
            voxel=voxels,
            num_trellis_steps=num_trellis_steps,
            trellis_cfg=trellis_cfg,
        )
        
        if self.save_gs_video and task_dir:
            torch.cuda.empty_cache()
            # Render the outputs
            video = render_utils.render_video(outputs[self.trellis_bake_decoder][0])["color"]
            imageio.mimsave(os.path.join(task_dir, f"{self.trellis_bake_decoder}_retexture.mp4"), video, fps=30)
        
        logger.info("[TRELLIS-PART] part texture generation finished in {:.3f}s".format(time.time() - tik))
        
        tik = time.time()
        # Convert to trimesh with texture
        textured_part = postprocessing_utils.to_trimesh(
            outputs[self.trellis_bake_decoder][0],
            part_mesh,
            vertices=part_mesh.vertices.astype(np.float32),
            faces=part_mesh.faces.astype(np.int32),
            # baking parameters
            texture_size=1024,
            texture_bake_mode=trellis_bake_mode,
            get_srgb_texture=get_srgb_texture,
            debug=False,
            verbose=False,
            forward_rot=False
        )
        # finally denormalise to make it compatible 
        textured_part = denormalise_mesh(textured_part, center, scale)
        
        logger.info("part postprocessing finished in {:.3f}s".format(time.time() - tik))
        
        savepath = ""
        if task_dir:
            retextured_parts_dir = os.path.join(task_dir, "retextured_parts")
            os.makedirs(retextured_parts_dir, exist_ok=True)
            savepath = os.path.join(retextured_parts_dir, f"retextured_part_{part_id:02d}.glb")
            textured_part.export(savepath)
            if self.verbose:
                textured_part.export(os.path.join(retextured_parts_dir, f"retextured_part_{part_id:02d}.obj"))
            logger.info(f"Re-textured part {part_id} saved at: {savepath}")
        
        return textured_part, savepath

    @timer_decorator
    def combine_retextured_part(
        self,
        full_textured_mesh: trimesh.Trimesh,
        retextured_part: trimesh.Trimesh,
        part_id: int,
        submesh_segment_indices: List[int],
        task_dir: str = "",
    ):
        """
        Combine a re-textured part back into the full mesh.
        
        Args:
            full_textured_mesh: The original full textured mesh
            retextured_part: The re-textured part mesh
            part_id: ID of the part that was re-textured
            submesh_segment_indices: Indices that define part boundaries
            task_dir: Directory to save results
        
        Returns:
            tuple: (combined_mesh, save_path)
        """
        logger.info(f"🔄 Combining re-textured part {part_id} with full mesh...")
        
        # Split the original mesh to get individual parts
        submeshes = self._split_textured_mesh(full_textured_mesh, submesh_segment_indices, task_dir)
        
        # Replace the specific part with the re-textured version
        if part_id < len(submeshes):
            submeshes[part_id] = retextured_part
            logger.info(f"✅ Replaced part {part_id} with re-textured version")
        else:
            raise ValueError(f"Part ID {part_id} is out of range. Available parts: 0-{len(submeshes)-1}")
        
        # Combine all parts into a single mesh
        # Create a scene to hold all submeshes
        combined_scene = trimesh.Scene()
        for i, submesh in enumerate(submeshes):
            combined_scene.add_geometry(submesh, node_name=f"part_{i:02d}")
        
        savepath = ""
        if task_dir:
            savepath = os.path.join(task_dir, f"combined_with_retextured_part_{part_id:02d}.glb")
            combined_scene.export(savepath)
            logger.info(f"Combined mesh with re-textured part {part_id} saved at: {savepath}")
        
        return combined_scene, savepath
