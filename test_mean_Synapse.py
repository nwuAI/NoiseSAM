import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from val import test_single_image_Synapse
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_ACDC import ACDC_dataset
import h5py
from icecream import ic
import pandas as pd
def inference(args, multimask_output, db_config, model, test_save_path=None):
    # 从 test.list 文件加载路径
    with open(os.path.join(args.root_path, 'test.list'), 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])

    if len(image_list) == 0:
        logging.error("No data found in test.list. Please check the file and its contents.")
        return

    logging.info("Starting inference...")

    # 初始化指标列表
    metric_list = []

    with torch.no_grad():
        for case in tqdm(image_list, ncols=70):
            # 加载图像和标签数据
            h5_path = os.path.join(args.root_path, "data", f"{case}.h5")
            if not os.path.exists(h5_path):
                logging.warning(f"File not found: {h5_path}. Skipping...")
                continue

            with h5py.File(h5_path, 'r') as h5f:
                image = h5f['image'][:]
                label = h5f['label'][:]

            # 转为 PyTorch 张量
            image = torch.from_numpy(image).unsqueeze(0).float().cuda()  # 添加批次维度
            label = torch.from_numpy(label).unsqueeze(0).float().cuda()  # 添加批次维度

            # 推理
            metric_i = test_single_image_Synapse(
                image=image,
                label=label,
                net=model,
                classes=args.num_classes + 1,
                multimask_output=multimask_output,
                patch_size=[args.img_size, args.img_size]
            )
            # 检查返回值是否为空
            if metric_i and isinstance(metric_i, list):
                metric_list.append(metric_i[0])  # 假设只计算第一个类别

    # 检查是否有有效的指标
    if len(metric_list) == 0:
        logging.error("No valid metrics were calculated. Please check your data and model.")
        return

    # 转换为 NumPy 数组
    metric_list = np.array(metric_list)

    # 确保指标数组形状正确
    if len(metric_list.shape) != 2 or metric_list.shape[1] != 4:
        raise ValueError(f"Invalid shape for metric_list: {metric_list.shape}, expected (N, 4).")

    # 计算平均指标
    avg_metric = np.mean(metric_list, axis=0)
    performance = avg_metric[0]  # 平均Dice
    mean_hd95 = avg_metric[1]    # 平均HD95
    mean_jc = avg_metric[2]      # 平均Jaccard系数
    mean_asd = avg_metric[3]     # 平均表面距离

    logging.info(f"Test Results - Dice: {performance:.4f}, HD95: {mean_hd95:.4f}, JC: {mean_jc:.4f}, ASD: {mean_asd:.4f}")
    print(f"Test Results - Dice: {performance:.4f}, HD95: {mean_hd95:.4f}, JC: {mean_jc:.4f}, ASD: {mean_asd:.4f}")

    # 保存结果
    if test_save_path is not None:
        save_csv = os.path.join(test_save_path, "test_metrics.csv")
        metrics_df = pd.DataFrame({
            "Dice": metric_list[:, 0],
            "HD95": metric_list[:, 1],
            "Jaccard": metric_list[:, 2],
            "ASD": metric_list[:, 3]
        })
        metrics_df.to_csv(save_csv, index=False)
        logging.info(f"Metrics saved to {save_csv}")

    return avg_metric


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                    default='', help='Name of Experiment')
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1337, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', default=False,help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='/media/amax/data/GL/CPC-SAM-main/SAM/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    
    parser.add_argument('--lora_ckpt', type=str, default='', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b_dualmask_same_prompt_class_random_large', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder_prompt')
    parser.add_argument('--exp', type=str, default='prediction_results.csv')
    parser.add_argument('--promptmode', type=str, default='point',help='prompt')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'num_classes': args.num_classes,
            'z_spacing': 8
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    
    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()
    #print(net)
    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path=test_save_path)

