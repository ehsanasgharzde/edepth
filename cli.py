import argparse
import logging
import sys
import torch
from edepth_rewrite.utils.config_loader import load_config
from edepth_rewrite.models.model_fixed import FixedDPTModel
from edepth_rewrite.training.trainer_fixed import Trainer
from edepth_rewrite.utils.export import export_onnx, export_torchscript, print_deployment_plan
from edepth_rewrite.datasets.nyu_dataset import NYUDepthV2Dataset
from edepth_rewrite.datasets.kitti_dataset import KITTIDepthDataset
import os

def main():
    parser = argparse.ArgumentParser(description="edepth SOTA CLI")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'inference', 'export_onnx', 'export_torchscript', 'deploy_plan'], required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path for resume/inference')
    parser.add_argument('--image', type=str, default=None, help='Image path for inference')
    parser.add_argument('--onnx_path', type=str, default='model.onnx')
    parser.add_argument('--ts_path', type=str, default='model.ts')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    device = args.device

    if args.mode == 'train':
        # Dataset selection
        if config['data']['dataset'] == 'nyu_depth_v2':
            dataset = NYUDepthV2Dataset(
                data_root=config['data']['data_root'],
                split='train',
                img_size=tuple(config['data']['img_size']),
                min_depth=config['data']['min_depth'],
                max_depth=config['data']['max_depth'],
                depth_scale=config['data']['depth_scale'],
                augmentation=config['data']['augmentation'],
                cache=False
            )
        elif config['data']['dataset'] == 'kitti':
            dataset = KITTIDepthDataset(
                data_root=config['data']['data_root'],
                split='train',
                img_size=tuple(config['data']['img_size']),
                min_depth=config['data']['min_depth'],
                max_depth=config['data']['max_depth'],
                depth_scale=config['data']['depth_scale'],
                augmentation=config['data']['augmentation'],
                cache=False
            )
        else:
            logging.error(f"Unknown dataset: {config['data']['dataset']}")
            sys.exit(1)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
        model = FixedDPTModel(
            backbone_name=config['model']['backbone'],
            extract_layers=config['model']['extract_layers'],
            decoder_channels=config['model']['decoder_channels'],
            patch_size=config['model']['patch_size'],
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained']
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
        from edepth_rewrite.losses.losses_fixed import SiLogLoss
        loss_fn = SiLogLoss()
        trainer = Trainer(model, loss_fn, optimizer, device=device, amp=config['training']['amp'], grad_clip=config['training']['grad_clip'])
        for epoch in range(config['training']['epochs']):
            loss = trainer.train_epoch(dataloader)
            trainer.save_checkpoint(epoch)
        logging.info("Training complete.")

    elif args.mode == 'eval':
        logging.info("Evaluation mode not yet implemented.")
        # Placeholder for evaluation logic

    elif args.mode == 'inference':
        from edepth_rewrite.inference.inference_fixed import run_inference
        if args.ckpt is None or args.image is None:
            logging.error("--ckpt and --image are required for inference mode.")
            sys.exit(1)
        model_kwargs = dict(
            backbone_name=config['model']['backbone'],
            extract_layers=config['model']['extract_layers'],
            decoder_channels=config['model']['decoder_channels'],
            patch_size=config['model']['patch_size'],
            num_classes=config['model']['num_classes'],
            pretrained=False
        )
        mean = config['data']['normalize_mean']
        std = config['data']['normalize_std']
        img_size = tuple(config['data']['img_size'])
        depth = run_inference(
            model_ckpt=args.ckpt,
            image=args.image,
            img_size=img_size,
            mean=mean,
            std=std,
            min_depth=config['data']['min_depth'],
            max_depth=config['data']['max_depth'],
            device=device,
            model_kwargs=model_kwargs
        )
        logging.info(f"Inference complete. Depth shape: {depth.shape}")

    elif args.mode == 'export_onnx':
        model = FixedDPTModel(
            backbone_name=config['model']['backbone'],
            extract_layers=config['model']['extract_layers'],
            decoder_channels=config['model']['decoder_channels'],
            patch_size=config['model']['patch_size'],
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained']
        ).to(device)
        dummy_input = torch.randn(1, 3, *config['data']['img_size']).to(device)
        export_onnx(model, dummy_input, args.onnx_path)

    elif args.mode == 'export_torchscript':
        model = FixedDPTModel(
            backbone_name=config['model']['backbone'],
            extract_layers=config['model']['extract_layers'],
            decoder_channels=config['model']['decoder_channels'],
            patch_size=config['model']['patch_size'],
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained']
        ).to(device)
        dummy_input = torch.randn(1, 3, *config['data']['img_size']).to(device)
        export_torchscript(model, dummy_input, args.ts_path)

    elif args.mode == 'deploy_plan':
        print_deployment_plan()

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 