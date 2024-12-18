import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, NuScenesDataset, create_nuscenes_info
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/bevfusion.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    root_path = Path(cfg.DATA_CONFIG.DATA_PATH)
    split_path = root_path / cfg.DATA_CONFIG.VERSION / cfg.DATA_CONFIG.INFO_PATH["test"][0]
    print("Split path: ", split_path)
    if not (split_path).exists():
        create_nuscenes_info(cfg.DATA_CONFIG.VERSION, root_path, root_path, max_sweeps=100, with_cam=True)
    logger.info(f"Path {split_path} exists...")
    demo_dataset = NuScenesDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, root_path=root_path, logger=logger, training=False)
    
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    #model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            #
            
            # Try to export onl the vfe
            voxels = data_dict["voxels"]
            voxel_num_points = data_dict["voxel_num_points"] 
            valid_indices = torch.arange(0, len(voxels))
            vfe = model.vfe

            torch.onnx.export(
                vfe,                  # model to export
                (voxels, voxel_num_points, valid_indices),        # inputs of the model,
                "vfe.onnx",        # filename of the ONNX model
                input_names=["voxels", "voxel_num_points", "valid_indices"],  # Rename inputs for the ONNX model
            )

            pred_dicts, _ = model.forward(data_dict)
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                #ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
