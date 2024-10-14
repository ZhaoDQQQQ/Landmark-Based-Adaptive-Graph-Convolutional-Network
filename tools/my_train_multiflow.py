# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import argparse
import os
import os.path as osp
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist, set_random_seed, load_checkpoint
from mmcv.utils import get_git_hash

from pyskl import __version__
from pyskl.apis import init_random_seed, train_model
from pyskl.datasets import build_dataset
from pyskl.models import build_model
from pyskl.utils import collect_env, get_root_logger, mc_off, mc_on, test_port, cache_checkpoint

os.environ['RANK'] = str(0)
os.environ['WORLD_SIZE'] = str(1)
os.environ['MASTER_ADDR'] = "localhost"
os.environ['MASTER_PORT'] = '22150'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    # 配置文件路径
    parser.add_argument(
        '-config',
        default=r'..\configs\oulu\time_seg_16\face_flow_4_j_b_jm_bm.py',
        help='train config file path')
    # 是否验证集
    parser.add_argument(
        '--validate',
        default='True',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    # 是否测试集
    parser.add_argument(
        '--test-last',
        default='True',
        action='store_true',
        help='whether to test the checkpoint after training')
    # 是否测试最佳checkpoint
    parser.add_argument(
        '--test-best',
        default='True',
        action='store_true',
        help='whether to test the best checkpoint (if applicable) after training')
    # 随机种子选择
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def run(ann_file, work_dir, flag):
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.work_dir = work_dir
    cfg.ann_file = ann_file
    cfg.data.train.dataset['ann_file'] = ann_file
    cfg.data.val.ann_file = ann_file
    cfg.data.test.ann_file = ann_file

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # config file > default (base filename)
    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='gloo')

    if flag:
        init_dist(args.launcher, **cfg.dist_params)
    else:
        pass

    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    auto_resume = cfg.get('auto_resume', True)
    if auto_resume and cfg.get('resume_from', None) is None:
        resume_pth = osp.join(cfg.work_dir, 'latest.pth')
        if osp.exists(resume_pth):
            cfg.resume_from = resume_pth

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    # env_info_dict = collect_env()
    # env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    # dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    # meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)

    cfg.seed = seed
    meta['seed'] = seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

    model = build_model(cfg.multiflow)

    flow_name = "model_"

    for flow in model.which_flow:
        model_temp = build_model(cfg[flow_name + flow])
        # checkpoint_temp = cache_checkpoint(cfg.flow_checkpoint[flow])
        # load_checkpoint(model_temp, checkpoint_temp, map_location='cpu')
        model.add_flow(model_temp, flow)

    x = list(model.named_parameters())

    datasets = [build_dataset(cfg.data.train)]

    cfg.workflow = cfg.get('workflow', [('train', 1)])
    assert len(cfg.workflow) == 1
    if cfg.checkpoint_config is not None:
        # save pyskl version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            pyskl_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text)

    test_option = dict(test_last=args.test_last, test_best=args.test_best)

    default_mc_cfg = ('localhost', 22077)
    memcached = cfg.get('memcached', False)

    if rank == 0 and memcached:
        # mc_list is a list of pickle files you want to cache in memory.
        # Basically, each pickle file is a dictionary.
        mc_cfg = cfg.get('mc_cfg', default_mc_cfg)
        assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
        if not test_port(mc_cfg[0], mc_cfg[1]):
            mc_on(port=mc_cfg[1], launcher=args.launcher)
        retry = 3
        while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
            time.sleep(5)
            retry -= 1
        assert retry >= 0, 'Failed to launch memcached. '

    dist.barrier()

    train_model(model, datasets, cfg, validate=args.validate, test=test_option, timestamp=timestamp, meta=meta)
    dist.barrier()

    if rank == 0 and memcached:
        mc_off()


def main():
    ann_file_0 = r"..\data\oulu_10_fold\oulu_0.pkl"
    work_dir_0 = r'..\work_dirs\\oulu_10_fold\16_segment\landmarks_0\j_b_jm_bm'
    flag = 1
    for i in range(0, 10):
        ann_file = ann_file_0.replace("oulu_0.pkl", "oulu_" + str(i) + ".pkl")
        work_dir = work_dir_0.replace("landmarks_0", "landmarks_" + str(i))
        run(ann_file, work_dir, flag)
        flag = 0


if __name__ == '__main__':
    main()

