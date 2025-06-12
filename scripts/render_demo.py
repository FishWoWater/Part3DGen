#!/usr/bin/env python3
"""
Enhanced Blender Demo Video Renderer for Part-Aware 3D Generation

This script creates a comprehensive demo video with three clips:
1. 360-degree rotation of raw geometry (normal map + pure geometry side-by-side)
2. 360-degree rotation of textured mesh
3. Part explosion and assembly animation

Usage:
    blender --background --python render_demo_enhanced.py -- --input_dir /path/to/task/output --output_dir /path/to/output
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

# Blender imports
import bpy
from mathutils import Vector


class PartAwareDemoRenderer:

    def __init__(self, input_dir: str, output_dir: str, width: int = 1024, height: int = 1024):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Animation settings
        self.clip_duration = 120  # 4 seconds at 30fps

        # Validate input directory structure
        self._validate_input_structure()

        # Animation parameters
        self.fps = 24
        self.rotation_duration = 3.0  # seconds for 360-degree rotation
        self.explosion_duration = 4.0  # seconds for part explosion/assembly
        self.frames_per_rotation = int(self.fps * self.rotation_duration)
        self.frames_explosion = int(self.fps * self.explosion_duration)

    def _validate_input_structure(self):
        """Validate that required files exist in input directory"""
        required_files = [
            self.input_dir / "raw_geometry.glb", self.input_dir / "textured_mesh.glb", self.input_dir / "textured_parts"
        ]

        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file/directory not found: {file_path}")

        # Check if textured parts exist
        part_files = list((self.input_dir / "textured_parts").glob("part_*.glb"))
        if not part_files:
            raise FileNotFoundError("No part files found in textured_parts directory")

        print(f"Found {len(part_files)} textured parts")

    def clear_scene(self):
        """Clear all objects from the scene"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Clear materials
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)

    def setup_render_settings(self, output_path: str, frame_start: int = 1, frame_end: int = None):
        """Setup render settings"""
        scene = bpy.context.scene

        # Set render engine to EEVEE for better performance
        try:
            scene.render.engine = 'BLENDER_EEVEE_NEXT'
        except:
            print("Warning: EEVEE_NEXT not available, using default EEVEE")
            scene.render.engine = 'BLENDER_EEVEE'
        # scene.render.engine = "CYCLES"

        # Set output settings
        scene.render.resolution_x = self.width
        scene.render.resolution_y = self.height
        scene.render.resolution_percentage = 100

        # Set frame range
        scene.frame_start = frame_start
        if frame_end:
            scene.frame_end = frame_end
        else:
            scene.frame_end = frame_start

        # Set output path and format
        scene.render.filepath = output_path
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
        scene.render.ffmpeg.constant_rate_factor = 'HIGH'

        # Set frame rate
        scene.render.fps = self.fps

    def setup_lighting_and_environment(self):
        """Setup basic lighting and environment"""
        # Enable world shader nodes
        world = bpy.context.scene.world
        world.use_nodes = True
        world_nodes = world.node_tree.nodes
        world_nodes.clear()

        # Create environment texture or basic color
        env_texture_node = world_nodes.new(type='ShaderNodeBackground')
        env_texture_node.inputs[0].default_value = (0.1, 0.1, 0.15, 1.0)  # Dark blue background
        env_texture_node.inputs[1].default_value = 1.0  # Strength

        world_output = world_nodes.new(type='ShaderNodeOutputWorld')
        world.node_tree.links.new(env_texture_node.outputs[0], world_output.inputs[0])

        # Add key light
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
        key_light = bpy.context.object
        key_light.data.energy = 3.0
        key_light.rotation_euler = (0.7, 0, 0.8)

        # Add fill light
        bpy.ops.object.light_add(type='AREA', location=(-3, -3, 5))
        fill_light = bpy.context.object
        fill_light.data.energy = 1.5
        fill_light.rotation_euler = (1.2, 0, -0.5)

    def load_and_prepare_mesh(self, mesh_path: str, name_prefix: str = ""):
        """Load and prepare a mesh file"""
        # Import the mesh
        if mesh_path.endswith('.glb') or mesh_path.endswith('.gltf'):
            bpy.ops.import_scene.gltf(filepath=mesh_path)
        elif mesh_path.endswith('.obj'):
            bpy.ops.import_scene.obj(filepath=mesh_path)
        else:
            raise ValueError(f"Unsupported file format: {mesh_path}")

        # Get the imported objects
        imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

        # Center and normalize the objects
        if imported_objects:
            self.normalize_objects(imported_objects)

        # Rename objects with prefix
        if name_prefix:
            for i, obj in enumerate(imported_objects):
                obj.name = f"{name_prefix}_{i:02d}"

        return imported_objects

    def normalize_objects(self, objects):
        """Normalize object positions and scale"""
        if not objects:
            return

        # Calculate bounding box of all objects
        all_vertices = []
        for obj in objects:
            if obj.type == 'MESH':
                mesh_vertices = [obj.matrix_world @ Vector(v.co) for v in obj.data.vertices]
                all_vertices.extend(mesh_vertices)

        if not all_vertices:
            return

        # Calculate center and scale
        min_coords = Vector(
            (min(v.x for v in all_vertices), min(v.y for v in all_vertices), min(v.z for v in all_vertices)))
        max_coords = Vector(
            (max(v.x for v in all_vertices), max(v.y for v in all_vertices), max(v.z for v in all_vertices)))

        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        max_dimension = max(size.x, size.y, size.z)

        # Normalize scale to fit in unit cube
        scale_factor = 1.8 / max_dimension if max_dimension > 0 else 1.0

        # Apply transformation to all objects
        for obj in objects:
            # Move to origin and scale
            obj.location = (obj.location - center) * scale_factor
            obj.scale = obj.scale * scale_factor

    def create_camera_path_360(self, radius: float = 3.0, height: float = 1.0):
        """Create a 360-degree camera rotation path"""
        # Create camera
        bpy.ops.object.camera_add(location=(radius, 0, height))
        camera = bpy.context.object

        # Point camera at origin
        direction = Vector((0, 0, 0)) - camera.location
        camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

        # Animate camera rotation
        for frame in range(self.frames_per_rotation):
            angle = (frame / self.frames_per_rotation) * 2 * math.pi
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            camera.location = (x, y, height)
            # Update rotation to look at center
            direction = Vector((0, 0, 0)) - camera.location
            camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

            camera.keyframe_insert(data_path="location", frame=frame + 1)
            camera.keyframe_insert(data_path="rotation_euler", frame=frame + 1)

        return camera

    def create_normal_material(self):
        """Create a material that shows normals as colors"""
        mat = bpy.data.materials.new(name="NormalVisualization")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()

        # Create nodes
        geometry_node = nodes.new(type='ShaderNodeNewGeometry')
        vector_transform = nodes.new(type='ShaderNodeVectorTransform')
        vector_transform.vector_type = 'NORMAL'
        vector_transform.convert_from = 'OBJECT'
        vector_transform.convert_to = 'CAMERA'

        # Convert from -1,1 to 0,1 range
        add_node = nodes.new(type='ShaderNodeVectorMath')
        add_node.operation = 'ADD'
        add_node.inputs[1].default_value = (1, 1, 1)

        multiply_node = nodes.new(type='ShaderNodeVectorMath')
        multiply_node.operation = 'MULTIPLY'
        multiply_node.inputs[1].default_value = (0.5, 0.5, 0.5)

        emission_node = nodes.new(type='ShaderNodeEmission')
        output_node = nodes.new(type='ShaderNodeOutputMaterial')

        # Connect nodes
        links = mat.node_tree.links
        links.new(geometry_node.outputs['Normal'], vector_transform.inputs['Vector'])
        links.new(vector_transform.outputs['Vector'], add_node.inputs[0])
        links.new(add_node.outputs['Vector'], multiply_node.inputs[0])
        links.new(multiply_node.outputs['Vector'], emission_node.inputs['Color'])
        links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

        return mat

    def render_clip_raw_geometry(self):
        """Render clip 1: Raw geometry with normal visualization"""
        print("Rendering Clip 1: Raw geometry 360-degree rotation...")

        # Render normal map version first
        self.clear_scene()
        self.setup_lighting_and_environment()

        # Load raw geometry
        raw_geo_path = str(self.input_dir / "raw_geometry.glb")
        objects = self.load_and_prepare_mesh(raw_geo_path, "raw_geo")

        # Apply normal material
        normal_material = self.create_normal_material()
        for obj in objects:
            obj.data.materials.clear()
            obj.data.materials.append(normal_material)

        # Setup camera
        camera = self.create_camera_path_360(radius=4.0, height=1.5)
        bpy.context.scene.camera = camera

        # Render normal map version
        normal_output_path = str(self.output_dir / "normal_temp.mp4")
        self.setup_render_settings(normal_output_path, 1, self.frames_per_rotation)
        bpy.ops.render.render(animation=True)

        # Render raw geometry version
        self.clear_scene()
        self.setup_lighting_and_environment()

        # Load raw geometry again
        objects = self.load_and_prepare_mesh(raw_geo_path, "raw_geo")

        # Setup camera again
        camera = self.create_camera_path_360(radius=4.0, height=1.5)
        bpy.context.scene.camera = camera

        # Render raw geometry version
        raw_output_path = str(self.output_dir / "raw_temp.mp4")
        self.setup_render_settings(raw_output_path, 1, self.frames_per_rotation)
        bpy.ops.render.render(animation=True)

        # Combine both videos: left half of normal map + right half of raw geometry
        final_output_path = str(self.output_dir / "raw_geometry.mp4")
        cmd = [
            'ffmpeg', '-i', normal_output_path, '-i', raw_output_path, '-filter_complex',
            f'[0:v]crop=w={self.width//2}:h={self.height}:x=0:y=0[left];[1:v]crop=w={self.width//2}:h={self.height}:x={self.width//2}:y=0[right];[left][right]hstack=inputs=2[v]',
            '-map', '[v]', '-c:v', 'libx264', '-y', final_output_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            # Clean up temporary files
            os.remove(normal_output_path)
            os.remove(raw_output_path)
            print(f"Combined normal map and raw geometry videos: {final_output_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error combining videos: {e}. Keeping separate files.")

        return final_output_path

    def render_clip_textured_mesh(self):
        """Render clip 2: Textured mesh 360-degree rotation"""
        print("Rendering Clip 2: Textured mesh 360-degree rotation...")

        self.clear_scene()
        self.setup_lighting_and_environment()

        # Load textured mesh
        textured_path = str(self.input_dir / "textured_mesh.glb")
        objects = self.load_and_prepare_mesh(textured_path, "textured")

        # Fix texture colorspace - textures are in RGB space and need gamma correction
        for obj in objects:
            if obj.type == 'MESH' and obj.data.materials:
                for material in obj.data.materials:
                    if material and material.use_nodes:
                        for node in material.node_tree.nodes:
                            if node.type == 'TEX_IMAGE' and node.image:
                                # Set colorspace to sRGB for proper gamma-corrected display
                                node.image.colorspace_settings.name = 'sRGB'

        # Setup camera
        camera = self.create_camera_path_360(radius=3.5, height=1.0)
        bpy.context.scene.camera = camera

        # Setup render settings
        output_path = str(self.output_dir / "textured_mesh.mp4")
        self.setup_render_settings(output_path, 1, self.frames_per_rotation)

        # Render
        bpy.ops.render.render(animation=True)

        return output_path

    def render_clips_part_explosion(self, part_dir: Path, clip_name: str):
        """Render clip 3: Part explosion and assembly animation"""
        print("Rendering Clip 3: Part explosion and assembly animation...")

        self.clear_scene()
        self.setup_lighting_and_environment()

        # Load all textured parts WITHOUT individual normalization
        # part_files = sorted(list((self.input_dir / "textured_parts").glob("part_*.glb")))
        part_files = sorted(list(Path(part_dir).glob("*part_*.glb")))
        if len(part_files) == 0:
            part_files = sorted(list(Path(part_dir).glob("*part_*.obj")))
        all_part_objects = []

        for i, part_file in enumerate(part_files):
            # Import the mesh without any automatic transformations
            if str(part_file).endswith('.glb') or str(part_file).endswith('.gltf'):
                bpy.ops.import_scene.gltf(filepath=str(part_file))
            elif str(part_file).endswith('.obj'):
                try:
                    bpy.ops.import_scene.obj(filepath=str(part_file))
                except:
                    bpy.ops.wm.obj_import(filepath=str(part_file))
            else:
                continue

            # Get the imported objects
            imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

            # Fix texture colorspace for consistent brightness with other clips
            for obj in imported_objects:
                if obj.type == 'MESH' and obj.data.materials:
                    for material in obj.data.materials:
                        if material and material.use_nodes:
                            for node in material.node_tree.nodes:
                                if node.type == 'TEX_IMAGE' and node.image:
                                    # Set colorspace to sRGB for proper gamma-corrected display
                                    node.image.colorspace_settings.name = 'sRGB'

            # Rename objects with prefix but do NOT normalize individually
            for j, obj in enumerate(imported_objects):
                obj.name = f"part_{i:02d}_{j:02d}"

            all_part_objects.extend(imported_objects)

        print(f"Loaded {len(all_part_objects)} part objects")

        # Calculate common normalization parameters for ALL parts together
        if all_part_objects:
            # Calculate bounding box of all parts together
            all_vertices = []
            for obj in all_part_objects:
                if obj.type == 'MESH':
                    mesh_vertices = [obj.matrix_world @ Vector(v.co) for v in obj.data.vertices]
                    all_vertices.extend(mesh_vertices)

            if all_vertices:
                # Calculate center and scale for all parts
                min_coords = Vector(
                    (min(v.x for v in all_vertices), min(v.y for v in all_vertices), min(v.z for v in all_vertices)))
                max_coords = Vector(
                    (max(v.x for v in all_vertices), max(v.y for v in all_vertices), max(v.z for v in all_vertices)))

                center = (min_coords + max_coords) / 2
                size = max_coords - min_coords
                max_dimension = max(size.x, size.y, size.z)

                # Common scale factor for all parts
                scale_factor = 1.8 / max_dimension if max_dimension > 0 else 1.0

                # Apply the SAME transformation to all parts
                for obj in all_part_objects:
                    # Move to origin and scale uniformly
                    obj.location = (obj.location - center) * scale_factor
                    obj.scale = obj.scale * scale_factor

        # Setup camera for explosion view
        bpy.ops.object.camera_add(location=(4, -4, 3))
        camera = bpy.context.object
        # Point camera at origin
        direction = Vector((0, 0, 0)) - camera.location
        camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        bpy.context.scene.camera = camera

        # Animation keyframes
        explosion_radius = 1.5
        assembly_start_frame = int(self.frames_explosion * 0.75)

        # Animate each part
        for i, obj in enumerate(all_part_objects):
            if not obj:
                continue

            # Store original position
            original_location = obj.location.copy()

            # Calculate explosion direction
            angle = (i / len(all_part_objects)) * 2 * math.pi
            explosion_direction = Vector((
                math.cos(angle),
                math.sin(angle),
                0.2 * (i % 3 - 1)  # Add vertical variation
            ))
            explosion_direction.normalize()
            explosion_target = original_location + explosion_direction * explosion_radius

            # Clear existing animation
            obj.animation_data_clear()

            # Keyframe 1: Original position
            obj.location = original_location
            obj.keyframe_insert(data_path="location", frame=1)
            # obj.keyframe_insert(data_path="rotation_euler", frame=1)

            # Keyframe 2: Exploded position
            obj.location = explosion_target
            obj.keyframe_insert(data_path="location", frame=assembly_start_frame)

            # Keyframe 3: Back to original
            obj.location = original_location
            obj.keyframe_insert(data_path="location", frame=self.frames_explosion)

            # Add rotation animation
            for frame in range(1, self.frames_explosion + 1):
                rotation_angle = (frame - 1) * 8 * math.pi / self.frames_explosion  # 4 full rotations
                obj.rotation_mode = "XYZ"
                obj.rotation_euler.z = rotation_angle
                obj.keyframe_insert(data_path="rotation_euler", frame=frame)

        # Setup render settings
        output_path = str(self.output_dir / f"{clip_name}.mp4")
        self.setup_render_settings(output_path, 1, self.frames_explosion)

        # Render
        bpy.ops.render.render(animation=True)

        return output_path

    def combine_clips_with_ffmpeg(self, clip_paths):
        """Combine clips using FFmpeg"""
        print("Combining clips into final video...")

        # Filter existing clips
        existing_clips = [path for path in clip_paths if os.path.exists(path)]

        if not existing_clips:
            print("No clips found to combine")
            return None

        if len(existing_clips) == 1:
            # If only one clip, create a cutscene and combine
            final_output = self.output_dir / "final_demo_video.mp4"
            clip_name = self._get_clip_name(existing_clips[0])
            cutscene_path = self._create_cutscene(clip_name, 0)

            # Create temporary clip list with cutscene
            clip_list_file = self.output_dir / "clip_list.txt"
            with open(clip_list_file, 'w') as f:
                f.write(f"file '{os.path.abspath(cutscene_path)}'\n")
                f.write(f"file '{os.path.abspath(existing_clips[0])}'\n")

            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i',
                str(clip_list_file), '-c', 'copy', '-y',
                str(final_output)
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                os.remove(cutscene_path)  # Clean up
                print(f"Final video created: {final_output}")
                return str(final_output)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Error creating final video: {e}")
                # Fallback to simple copy
                import shutil
                shutil.copy2(existing_clips[0], final_output)
                return str(final_output)

        # Create cutscenes for each clip
        cutscenes = []
        for i, clip_path in enumerate(existing_clips):
            clip_name = self._get_clip_name(clip_path)
            cutscene_path = self._create_cutscene(clip_name, i)
            if cutscene_path:
                cutscenes.append(cutscene_path)

        # Create clip list file for FFmpeg with cutscenes
        clip_list_file = self.output_dir / "clip_list.txt"
        with open(clip_list_file, 'w') as f:
            for i, clip_path in enumerate(existing_clips):
                if i < len(cutscenes):
                    f.write(f"file '{os.path.abspath(cutscenes[i])}'\n")
                f.write(f"file '{os.path.abspath(clip_path)}'\n")

        final_output = self.output_dir / "final_demo_video.mp4"

        # FFmpeg command to concatenate
        cmd = [
            'ffmpeg',
            '-f',
            'concat',
            '-safe',
            '0',
            '-i',
            str(clip_list_file),
            '-c',
            'copy',
            '-y',  # Overwrite output
            str(final_output)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            # Clean up cutscene files
            for cutscene in cutscenes:
                if os.path.exists(cutscene):
                    os.remove(cutscene)
            print(f"Final video created: {final_output}")
            return str(final_output)
        except subprocess.CalledProcessError as e:
            print(f"Error combining clips with FFmpeg: {e}")
            print(f"stderr: {e.stderr}")
            return None
        except FileNotFoundError:
            print("FFmpeg not found. Install FFmpeg to combine clips.")
            return None

    def _get_clip_name(self, clip_path):
        """Extract a human-readable name from clip path"""
        filename = os.path.basename(clip_path)
        if "raw_geometry" in filename:
            return "Generated Geometry"
        elif "raw_part_explosion" in filename:
            return "Generated Parts"
        elif "completed_part_explosion" in filename:
            return "Completed Parts"
        elif "textured_mesh" in filename:
            return "Textured Mesh"
        elif "textured_part_explosion" in filename:
            return "Textured Parts"
        elif "retextured_part_explosion" in filename:
            return "Re-textured Parts"
        elif "retexture_parts" in filename:
            # return "Part Retexture   " + '_'.join(os.path.splitext(filename)[0].split('_')[2:]).replace(
            #     "__", ", ").replace("_", " ")
            return "Part Retexture"
        else:
            return filename
            # return "Animation Clip"

    def _create_cutscene(self, text, index):
        """Create a 1-second cutscene with text on black background"""
        cutscene_path = str(self.output_dir / f"cutscene_{index}.mp4")

        # Use drawtext filter to create text on black background
        cmd = [
            'ffmpeg', '-f', 'lavfi', '-i', f'color=c=black:size={self.width}x{self.height}:duration=1:rate={self.fps}',
            '-vf', f"drawtext=text='{text}':fontsize=72:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2", '-c:v',
            'libx264', '-y', cutscene_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return cutscene_path
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error creating cutscene: {e}")
            return None

    def render_demo(self, num_parts: int = 3):
        """Render complete demo video"""
        print(f"Starting demo render for: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")

        # if os.path.exists(os.path.join(self.output_dir, "final_demo_video.mp4")):
        #     print("Final demo video already exists. Skipping render.")
        #     return

        clip_paths = []

        try:
            # Render clip 1: Raw geometry
            clip_raw_geometry = self.render_clip_raw_geometry()
            if os.path.exists(clip_raw_geometry):
                clip_paths.append(clip_raw_geometry)

            # Render clip 2: explosion of the raw parts
            clip_raw_explosion = self.render_clips_part_explosion(self.input_dir / "raw_parts" / str(num_parts),
                                                                  "raw_part_explosion")
            if os.path.exists(clip_raw_explosion):
                clip_paths.append(clip_raw_explosion)

            # Render clip 3: explosion of the completed parts
            if Path(self.input_dir / "completed_parts").exists():
                clip_completed_explosion = self.render_clips_part_explosion(self.input_dir / "completed_parts",
                                                                            "completed_part_explosion")
                if os.path.exists(clip_completed_explosion):
                    clip_paths.append(clip_completed_explosion)

            # Render clip 4: Textured mesh
            # clip_textured_mesh = self.render_clip_textured_mesh()
            # if os.path.exists(clip_textured_mesh):
            #     clip_paths.append(clip_textured_mesh)

            # Render clip 5: Part explosion of the textured parts
            clip_textured_explosion = self.render_clips_part_explosion(self.input_dir / "textured_parts",
                                                                       "textured_part_explosion")
            if os.path.exists(clip_textured_explosion):
                clip_paths.append(clip_textured_explosion)

            # Render clip 6+: Retextured parts explosion if available
            retextured_dirs = self.input_dir.glob("retexture_parts*")
            if retextured_dirs:
                # If retextured parts exist, render them
                for retextured_dir in retextured_dirs:
                    clip_retextured_explosion = self.render_clips_part_explosion(retextured_dir,
                                                                                 Path(retextured_dir).name)
                    if os.path.exists(clip_retextured_explosion):
                        clip_paths.append(clip_retextured_explosion)

            # Combine clips
            final_video = self.combine_clips_with_ffmpeg(clip_paths)

            print("\n" + "=" * 50)
            print("DEMO RENDER COMPLETE!")
            print("=" * 50)
            print("Individual clips:")
            for i, clip_path in enumerate(clip_paths, 1):
                print(f"  Clip {i}: {clip_path}")

            if final_video:
                print(f"\nFinal combined video: {final_video}")

        except Exception as e:
            print(f"Error during rendering: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    # Parse command line arguments (handling Blender's argument structure)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(description="Render demo video for part-aware 3D generation")
    parser.add_argument("--input-dir", required=True, help="Input directory containing the generated 3D models")
    parser.add_argument("--output-dir", required=True, help="Output directory for rendered videos")
    parser.add_argument("--width", type=int, default=1024, help="Video width")
    parser.add_argument("--height", type=int, default=1024, help="Video height")
    parser.add_argument("--num-parts", type=int, default=3, help="Number of parts to explode in the demo")

    args = parser.parse_args(argv)

    # Create renderer and run demo
    renderer = PartAwareDemoRenderer(input_dir=os.path.abspath(args.input_dir),
                                     output_dir=os.path.abspath(args.output_dir),
                                     width=args.width,
                                     height=args.height)

    renderer.render_demo(num_parts=args.num_parts)


if __name__ == "__main__":
    main()
