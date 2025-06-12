import logging
import os
import random
import shutil

import numpy as np
import torch
from easydict import EasyDict as edict
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from partfield.config import setup
from partfield.utils import load_mesh_util
from sklearn.cluster import AgglomerativeClustering, KMeans

from .partfield_utils import (
    construct_face_adjacency_matrix_ccmst,
    construct_face_adjacency_matrix_facemst,
    construct_face_adjacency_matrix_naive,
    export_colored_mesh_ply,
    hierarchical_clustering_labels,
)

PARTFIELD_PREFIX = "exp_results/"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PartFieldRunner:
    def __init__(self, config_file : str = "thirdparty/PartField/configs/final/demo.yaml", 
                 continue_ckpt: str = "pretrained/PartField/model_objaverse.pt", 
                 use_agglo: bool = True, alg_option: int = 0, 
                 with_knn: bool = False, export_mesh: bool = False):
        self.config_file = config_file
        self.continue_ckpt = continue_ckpt

        self.use_agglo = use_agglo
        self.alg_option = alg_option
        self.with_knn = with_knn 
        self.export_mesh = export_mesh

        args = edict({
            "config_file": self.config_file, 
            "opts": ["continue_ckpt", self.continue_ckpt]
        })
        cfg = setup(args, freeze=False)
    
        seed_everything(cfg.seed)

        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        
        checkpoint_callbacks = [ModelCheckpoint(
            monitor="train/current_epoch",
            dirpath=cfg.output_dir,
            filename="{epoch:02d}",
            save_top_k=100,
            save_last=True,
            every_n_epochs=cfg.save_every_epoch,
            mode="max",
            verbose=True
        )]

        self.trainer = Trainer(devices=-1,
                        accelerator="gpu",
                        precision="16-mixed",
                        strategy=DDPStrategy(find_unused_parameters=True),
                        max_epochs=cfg.training_epochs,
                        log_every_n_steps=1,
                        limit_train_batches=3500,
                        limit_val_batches=None,
                        callbacks=checkpoint_callbacks
                        )     
        self.cfg = cfg
        
        
    def run_partfield(self, mesh_path: str, feature_dir: str, cluster_dir: str, export_mesh: bool = False, num_max_clusters: int = 10):
        # here mesh_path is expected in the form of "exp_results/pipeline/<task_uid>/raw_geometry.glb"
        # feature dir is expected in the form of "exp_results/pipeline/<task_uid>/partfield_features/"
        # cluster_dir is expected in the form of "exp_results/pipeline/<task_uid>/clustering/"
        assert feature_dir.startswith(PARTFIELD_PREFIX), "feature_dir should start with 'exp_results/'"
        assert cluster_dir.startswith(PARTFIELD_PREFIX), "cluster_dir should start with 'exp_results/'"
        cluster_subfolder = os.path.join(cluster_dir, "cluster_out")
        os.makedirs(cluster_subfolder, exist_ok=True) 

        if export_mesh:
            # for mesh exportation
            ply_subfolder = os.path.join(cluster_dir, "ply")
            os.makedirs(ply_subfolder, exist_ok=True)    
            
        # Step1, first make a temp directory to store only the input mesh, and update the trainer 
        # TODO: optimize the running speed here by using pure pytorch inference 
        mesh_name, mesh_ext = os.path.splitext(os.path.basename(mesh_path))
        # temp directory for running PartField: "exp_results/pipeline/<task_uid>/raw_geometry/"
        temp_dir = os.path.join(os.path.dirname(mesh_path), mesh_name)
        os.makedirs(temp_dir, exist_ok=True)
        if os.path.exists(os.path.join(temp_dir, mesh_name + mesh_ext)):
            os.remove(os.path.join(temp_dir, mesh_name + mesh_ext))
        shutil.copy(mesh_path, os.path.join(temp_dir, mesh_name + mesh_ext))
        
        # update the data path 
        self.cfg.dataset.data_path = temp_dir
        # update the saving directory 
        self.cfg.result_name = os.path.join(feature_dir[len(PARTFIELD_PREFIX):], mesh_name)
        from partfield.model_trainer_pvcnn_only_demo import Model
        self.model = Model(self.cfg)
        self.trainer.predict(self.model, ckpt_path=self.cfg.continue_ckpt)
        # when finished, run clustering 
        return self.solve_clustering(os.path.join(temp_dir, mesh_name + mesh_ext), feature_dir, cluster_dir, num_max_clusters=num_max_clusters, export_mesh=export_mesh)
    
    def solve_clustering(self, model_path: str, feature_dir: str, cluster_dir: str, num_max_clusters: int = 10, export_mesh: bool = False):
        input_fname = model_path
        view_id = 0
        model_name = os.path.basename(input_fname).split('.')[0]
        mesh = load_mesh_util(input_fname)

        ### Load inferred PartField features
        try:
            point_feat = np.load(f'{feature_dir}/{model_name}/part_feat_{model_name}_{view_id}.npy')
        except:
            try:
                point_feat = np.load(f'{feature_dir}/{model_name}/part_feat_{model_name}_{view_id}_batch.npy')

            except:
                logger.error("pointfeat loading error. skipping...")
                logger.error(f'{feature_dir}/{model_name}/part_feat_{model_name}_{view_id}_batch.npy')
                return

        point_feat = point_feat / np.linalg.norm(point_feat, axis=-1, keepdims=True)

        if not self.use_agglo:
            for num_cluster in range(2, num_max_clusters):
                clustering = KMeans(n_clusters=num_cluster, random_state=0).fit(point_feat)
                labels = clustering.labels_

                pred_labels = np.zeros((len(labels), 1))
                for i, label in enumerate(np.unique(labels)):
                    # print(i, label)
                    pred_labels[labels == label] = i  # Assign RGB values to each label

                fname_clustering = os.path.join(cluster_dir, "cluster_out", str(model_name) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2))
                np.save(fname_clustering, pred_labels)
                
                V = mesh.vertices
                F = mesh.faces

                if export_mesh :
                    fname_mesh = os.path.join(cluster_dir, "ply", str(model_name) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2) + ".ply")
                    export_colored_mesh_ply(V, F, pred_labels, filename=fname_mesh)
            return pred_labels
        else:
            if self.alg_option == 0:
                adj_matrix = construct_face_adjacency_matrix_naive(mesh.faces)
            elif self.alg_option == 1:
                adj_matrix = construct_face_adjacency_matrix_facemst(mesh.faces, mesh.vertices, with_knn=self.with_knn)
            else:
                adj_matrix = construct_face_adjacency_matrix_ccmst(mesh.faces, mesh.vertices, with_knn=self.with_knn)

            clustering = AgglomerativeClustering(connectivity=adj_matrix,
                                        n_clusters=1,
                                        ).fit(point_feat)
            hierarchical_labels = hierarchical_clustering_labels(clustering.children_, point_feat.shape[0], max_cluster=num_max_clusters)

            all_FL = []
            for n_cluster in range(num_max_clusters):
                logger.debug("Processing cluster: "+str(n_cluster))
                labels = hierarchical_labels[n_cluster]
                all_FL.append(labels)
            
            all_FL = np.array(all_FL)
            unique_labels = np.unique(all_FL)

            for n_cluster in range(num_max_clusters):
                FL = all_FL[n_cluster]
                relabel = np.zeros((len(FL), 1))
                for i, label in enumerate(unique_labels):
                    relabel[FL == label] = i  # Assign RGB values to each label

                V = mesh.vertices
                F = mesh.faces

                if export_mesh :
                    fname_mesh = os.path.join(cluster_dir, "ply", str(model_name) + "_" + str(view_id) + "_" + str(num_max_clusters - n_cluster).zfill(2) + ".ply")
                    export_colored_mesh_ply(V, F, FL, filename=fname_mesh)

                fname_clustering = os.path.join(cluster_dir, "cluster_out", str(model_name) + "_" + str(view_id) + "_" + str(num_max_clusters - n_cluster).zfill(2))
                np.save(fname_clustering, FL)
        
            return hierarchical_labels
            
