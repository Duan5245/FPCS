"""
Created on September 5th, 2018
Modified and merged on [Current Date]

@author: itailang
"""

# import system modules
import os.path as osp
import sys
import argparse
import numpy as np

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from reconstruction.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from reconstruction.src.autoencoder import Configuration as Conf
from reconstruction.src.point_net_ae import PointNetAutoEncoder
from reconstruction.src.samplers import sampler_with_convs_and_symmetry_and_fc
from reconstruction.src.progressive_net_point_net_ae import ProgressiveNetPointNetAutoEncoder
from reconstruction.src.s_net_point_net_ae import SNetPointNetAutoEncoder
from reconstruction.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
    load_and_split_all_point_clouds_under_folder
from reconstruction.src.tf_utils import reset_tf_graph
from reconstruction.src.general_utils import plot_3d_point_cloud


# 主函数
def main():
    parser = argparse.ArgumentParser(description='Point Cloud Autoencoder and Samplers')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # 自动编码器训练命令
    train_ae_parser = subparsers.add_parser('train_ae', help='Train autoencoder')
    train_ae_parser.add_argument('--use_fps', type=bool, default=False,
                                 help='FPS sampling before autoencoder [default: False]')
    train_ae_parser.add_argument('--n_sample_points', type=int, default=2048,
                                 help='Number of sample points [default: 2048]')
    train_ae_parser.add_argument('--object_class', type=str, default='multi',
                                 help='Class name or multi [default: multi]')
    train_ae_parser.add_argument('--train_folder', type=str, default='log/autoencoder',
                                 help='Training folder [default: log/autoencoder]')

    # 自动编码器评估命令
    eval_ae_parser = subparsers.add_parser('eval_ae', help='Evaluate autoencoder')
    eval_ae_parser.add_argument('--use_fps', type=int, default=0, help='Use FPS sampling [default: 0]')
    eval_ae_parser.add_argument('--n_sample_points', type=int, default=2048,
                                help='Number of sample points [default: 2048]')
    eval_ae_parser.add_argument('--object_class', type=str, default='multi',
                                help='Class name or multi [default: multi]')
    eval_ae_parser.add_argument('--train_folder', type=str, default='log/autoencoder',
                                help='Training folder [default: log/autoencoder]')
    eval_ae_parser.add_argument('--restore_epoch', type=int, default=500, help='Restore epoch [default: 500]')
    eval_ae_parser.add_argument('--visualize_results', action='store_true', help='Visualize results [default: False]')

    # 渐进式网络训练命令
    train_prog_parser = subparsers.add_parser('train_prog', help='Train progressive network')
    train_prog_parser.add_argument('--n_sample_points', type=int, default=64,
                                   help='Number of sample points [default: 64]')
    train_prog_parser.add_argument('--similarity_reg_weight', type=float, default=0.01,
                                   help='Similarity regularization weight [default: 0.01]')
    train_prog_parser.add_argument('--learning_rate', type=float, default=0.0005,
                                   help='Learning rate [default: 0.0005]')
    train_prog_parser.add_argument('--restore_ae', type=bool, default=True, help='Restore trained AE [default: True]')
    train_prog_parser.add_argument('--fixed_ae', type=bool, default=True, help='Fixed AE model [default: True]')
    train_prog_parser.add_argument('--object_class', type=str, default='multi',
                                   help='Class name or multi [default: multi]')
    train_prog_parser.add_argument('--ae_folder', type=str, default='log/autoencoder',
                                   help='AE folder [default: log/autoencoder]')
    train_prog_parser.add_argument('--train_folder', type=str, default='log/progressive_net',
                                   help='Training folder [default: log/progressive_net]')

    # 渐进式网络评估命令
    eval_prog_parser = subparsers.add_parser('eval_prog', help='Evaluate progressive network')
    eval_prog_parser.add_argument('--n_sample_points', type=int, default=64,
                                  help='Number of sample points [default: 64]')
    eval_prog_parser.add_argument('--object_class', type=str, default='multi',
                                  help='Class name or multi [default: multi]')
    eval_prog_parser.add_argument('--train_folder', type=str, default='log/progressive_net',
                                  help='Training folder [default: log/progressive_net]')
    eval_prog_parser.add_argument('--visualize_results', action='store_true', help='Visualize results [default: False]')

    # S-Net训练命令
    train_snet_parser = subparsers.add_parser('train_snet', help='Train S-Net')
    train_snet_parser.add_argument('--n_sample_points', type=int, default=64,
                                   help='Number of sample points [default: 64]')
    train_snet_parser.add_argument('--similarity_reg_weight', type=float, default=0.01,
                                   help='Similarity regularization weight [default: 0.01]')
    train_snet_parser.add_argument('--learning_rate', type=float, default=0.0005,
                                   help='Learning rate [default: 0.0005]')
    train_snet_parser.add_argument('--restore_ae', type=bool, default=True, help='Restore trained AE [default: True]')
    train_snet_parser.add_argument('--fixed_ae', type=bool, default=True, help='Fixed AE model [default: True]')
    train_snet_parser.add_argument('--object_class', type=str, default='multi',
                                   help='Class name or multi [default: multi]')
    train_snet_parser.add_argument('--ae_folder', type=str, default='log/autoencoder',
                                   help='AE folder [default: log/autoencoder]')
    train_snet_parser.add_argument('--train_folder', type=str, default='log/s_net',
                                   help='Training folder [default: log/s_net]')

    # S-Net评估命令
    eval_snet_parser = subparsers.add_parser('eval_snet', help='Evaluate S-Net')
    eval_snet_parser.add_argument('--object_class', type=str, default='multi',
                                  help='Class name or multi [default: multi]')
    eval_snet_parser.add_argument('--train_folder', type=str, default='log/s_net',
                                  help='Training folder [default: log/s_net]')
    eval_snet_parser.add_argument('--visualize_results', action='store_true', help='Visualize results [default: False]')

    args = parser.parse_args()
    print(f"Command: {args.command}, Flags: {args}")

    project_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    top_in_dir = osp.join(project_dir, 'data', 'shape_net_core_uniform_samples_2048')
    top_out_dir = osp.join(project_dir, 'results')

    if args.object_class == 'multi':
        class_name = ['chair', 'table', 'car', 'airplane']
    else:
        class_name = [str(args.object_class)]

    # 加载点云数据
    def load_point_clouds():
        syn_id = snc_category_to_synth_id()[class_name[0]]
        class_dir = osp.join(top_in_dir, syn_id)
        pc_data_train, pc_data_val, pc_data_test = load_and_split_all_point_clouds_under_folder(
            class_dir, n_threads=8, file_ending='.ply', verbose=True
        )

        for i in range(1, len(class_name)):
            syn_id = snc_category_to_synth_id()[class_name[i]]
            class_dir = osp.join(top_in_dir, syn_id)
            pc_data_train_curr, pc_data_val_curr, pc_data_test_curr = load_and_split_all_point_clouds_under_folder(
                class_dir, n_threads=8, file_ending='.ply', verbose=True
            )
            pc_data_train.merge(pc_data_train_curr)
            pc_data_val.merge(pc_data_val_curr)
            pc_data_test.merge(pc_data_test_curr)

        if args.object_class == 'multi':
            pc_data_train.shuffle_data(seed=55)
            pc_data_val.shuffle_data(seed=55)
            pc_data_test.shuffle_data(seed=55)

        return pc_data_train, pc_data_val, pc_data_test

    # 执行不同命令
    if args.command == 'train_ae':
        pc_data_train, pc_data_val, _ = load_point_clouds()
        train_autoencoder(args, pc_data_train, pc_data_val, top_out_dir)

    elif args.command == 'eval_ae':
        _, _, pc_data_test = load_point_clouds()
        evaluate_autoencoder(args, pc_data_test, top_out_dir)

    elif args.command == 'train_prog':
        pc_data_train, pc_data_val, _ = load_point_clouds()
        train_progressive_net(args, pc_data_train, pc_data_val, top_out_dir)

    elif args.command == 'eval_prog':
        _, _, pc_data_test = load_point_clouds()
        evaluate_progressive_net(args, pc_data_test, top_out_dir)

    elif args.command == 'train_snet':
        pc_data_train, pc_data_val, _ = load_point_clouds()
        train_snet(args, pc_data_train, pc_data_val, top_out_dir)

    elif args.command == 'eval_snet':
        _, _, pc_data_test = load_point_clouds()
        evaluate_snet(args, pc_data_test, top_out_dir)


# 自动编码器训练函数
def train_autoencoder(args, pc_data_train, pc_data_val, top_out_dir):
    n_pc_points = 2048
    bneck_size = 128
    ae_loss = 'chamfer'
    experiment_name = 'autoencoder'

    # 加载默认训练参数
    train_params = default_train_params()
    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)

    train_dir = create_dir(osp.join(top_out_dir, args.train_folder))

    conf = Conf(
        n_input=[n_pc_points, 3],
        loss=ae_loss,
        training_epochs=train_params['training_epochs'],
        batch_size=train_params['batch_size'],
        denoising=train_params['denoising'],
        learning_rate=train_params['learning_rate'],
        train_dir=train_dir,
        loss_display_step=train_params['loss_display_step'],
        saver_step=train_params['saver_step'],
        z_rotate=train_params['z_rotate'],
        encoder=encoder,
        decoder=decoder,
        encoder_args=enc_args,
        decoder_args=dec_args
    )
    conf.experiment_name = experiment_name
    conf.held_out_step = 5
    conf.class_name = [args.object_class]
    conf.use_fps = args.use_fps
    conf.n_sample_points = args.n_sample_points
    conf.n_samp_out = [2048, 3]
    conf.save(osp.join(train_dir, 'configuration'))

    # 构建AE模型
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)

    # 训练AE
    buf_size = 1
    with open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size) as fout:
        train_stats = ae.train(pc_data_train, conf, log_file=fout, held_out_data=pc_data_val)


# 自动编码器评估函数
def evaluate_autoencoder(args, pc_data_test, top_out_dir):
    train_dir = create_dir(osp.join(top_out_dir, args.train_folder))
    conf = Conf.load(osp.join(train_dir, 'configuration'))
    conf.use_fps = args.use_fps
    conf.n_sample_points = args.n_sample_points

    # 构建AE模型
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)

    # 恢复保存的模型
    ae.restore_model(train_dir, epoch=args.restore_epoch, verbose=True)

    n_sample_points = args.n_sample_points
    eval_dir = create_dir(osp.join(train_dir, "eval"))

    # FPS采样
    if args.use_fps:
        sampled_pc, sample_idx = ae.get_samples(pc_data_test.point_clouds)
        np.save(osp.join(eval_dir, f"sampled_pc_test_set_{args.object_class}_fps_{n_sample_points:04d}.npy"),
                sampled_pc)
        np.save(osp.join(eval_dir, f"sample_idx_test_set_{args.object_class}_fps_{n_sample_points:04d}.npy"),
                sample_idx)

    # 重建点云
    reconstructions = ae.get_reconstructions(pc_data_test.point_clouds)
    suffix = f"_fps_{n_sample_points:04d}" if args.use_fps else ""
    np.save(osp.join(eval_dir, f"reconstructions_test_set_{args.object_class}{suffix}.npy"), reconstructions)

    # 计算损失
    loss_per_pc = ae.get_loss_per_pc(pc_data_test.point_clouds)
    np.save(osp.join(eval_dir, f"ae_loss_test_set_{args.object_class}{suffix}.npy"), loss_per_pc)

    # 保存日志
    log_suffix = f"_fps_{n_sample_points:04d}" if args.use_fps else ""
    with open(osp.join(eval_dir, f"eval_stats_test_set_{args.object_class}{log_suffix}.txt"), "w", 1) as log_file:
        log_file.write(f"Mean ae loss: {loss_per_pc.mean():.9f}\n")

        # 计算标准化重建误差
        ref_path = osp.join(eval_dir, f"ae_loss_test_set_{args.object_class}.npy")
        if osp.exists(ref_path):
            loss_per_pc_ref = np.load(ref_path)
            nre_per_pc = np.divide(loss_per_pc, loss_per_pc_ref)
            log_file.write(f"Normalized reconstruction error: {nre_per_pc.mean():.3f}\n")

    # 可视化结果
    if args.visualize_results:
        i = 0
        pc = pc_data_test.point_clouds[i]
        plot_3d_point_cloud(pc[:, 0], pc[:, 1], pc[:, 2], in_u_sphere=True, title="Complete input point cloud")

        if args.use_fps:
            sampled = sampled_pc[i]
            plot_3d_point_cloud(sampled[:, 0], sampled[:, 1], sampled[:, 2], in_u_sphere=True,
                                title="FPS sampled points")

        recon = reconstructions[i]
        recon_title = "Reconstruction from FPS points" if args.use_fps else "Reconstruction from complete input"
        plot_3d_point_cloud(recon[:, 0], recon[:, 1], recon[:, 2], in_u_sphere=True, title=recon_title)


# 渐进式网络训练函数
def train_progressive_net(args, pc_data_train, pc_data_val, top_out_dir):
    ae_dir = osp.join(top_out_dir, args.ae_folder)
    conf = Conf.load(osp.join(ae_dir, 'configuration'))

    # 更新配置
    conf.ae_dir = ae_dir
    conf.ae_name = 'autoencoder'
    conf.restore_ae = args.restore_ae
    conf.ae_restore_epoch = 500
    conf.fixed_ae = args.fixed_ae

    if conf.fixed_ae:
        conf.encoder_args['b_norm_decay'] = 1.0
        conf.decoder_args['b_norm_decay'] = 1.0
        conf.decoder_args['b_norm_decay_finish'] = 1.0

    # 采样器配置
    conf.experiment_name = 'sampler'
    conf.pc_size = [2 ** i for i in range(4, 12)]
    conf.n_samp = [args.n_sample_points, 3]
    conf.sampler = sampler_with_convs_and_symmetry_and_fc
    conf.similarity_reg_weight = args.similarity_reg_weight
    conf.learning_rate = args.learning_rate

    train_dir = create_dir(osp.join(top_out_dir, args.train_folder))
    conf.train_dir = train_dir
    conf.save(osp.join(train_dir, 'configuration'))

    # 构建模型
    reset_tf_graph()
    ae = ProgressiveNetPointNetAutoEncoder(conf.experiment_name, conf)

    # 训练采样器
    buf_size = 1
    with open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size) as fout:
        train_stats = ae.train(pc_data_train, conf, log_file=fout, held_out_data=pc_data_val)


# 渐进式网络评估函数
def evaluate_progressive_net(args, pc_data_test, top_out_dir):
    train_dir = osp.join(top_out_dir, args.train_folder)
    restore_epoch = 500
    conf = Conf.load(osp.join(train_dir, 'configuration'))

    conf.pc_size = [args.n_sample_points]
    conf.n_samp = [args.n_sample_points, 3]

    # 恢复模型
    reset_tf_graph()
    ae = ProgressiveNetPointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(train_dir, epoch=restore_epoch, verbose=True)

    n_input_points = conf.n_input[0]
    n_sample_points = conf.n_samp[0]
    eval_dir = create_dir(osp.join(train_dir, "eval"))

    # 采样点云
    file_path = osp.join(eval_dir, f"sampled_pc_test_set_{args.object_class}_{n_input_points:04d}.npy")
    if not osp.exists(file_path):
        _, sampled_pc, sample_idx, _ = ae.get_samples(pc_data_test.point_clouds)
        np.save(file_path, sampled_pc)
        np.save(osp.join(eval_dir, f"sample_idx_test_set_{args.object_class}_{n_input_points:04d}.npy"), sample_idx)
    else:
        sampled_pc = np.load(file_path)

    # 重建点云
    reconstructions = ae.get_reconstructions_from_sampled(sampled_pc)
    np.save(osp.join(eval_dir, f"reconstructions_test_set_{args.object_class}_{n_sample_points:04d}.npy"),
            reconstructions)

    # 计算损失
    ae_loss_per_pc = ae.get_loss_ae_per_pc(pc_data_test.point_clouds, sampled_pc)
    np.save(osp.join(eval_dir, f"ae_loss_test_set_{args.object_class}_{n_sample_points:04d}.npy"), ae_loss_per_pc)

    # 保存日志
    with open(osp.join(eval_dir, f"eval_stats_test_set_{args.object_class}_{n_sample_points:04d}.txt"), "w",
              1) as log_file:
        log_file.write(f"Evaluation flags: {args}\n")
        log_file.write(f"Mean ae loss: {ae_loss_per_pc.mean():.9f}\n")

        # 计算标准化重建误差
        ref_path = osp.join(conf.ae_dir, "eval", f"ae_loss_test_set_{args.object_class}.npy")
        if osp.exists(ref_path):
            ae_loss_per_pc_ref = np.load(ref_path)
            nre_per_pc = np.divide(ae_loss_per_pc, ae_loss_per_pc_ref)
            log_file.write(f"Normalized reconstruction error: {nre_per_pc.mean():.9f}\n")

    # 可视化结果
    if args.visualize_results:
        i = 0
        pc = pc_data_test.point_clouds[i]
        plot_3d_point_cloud(pc[:, 0], pc[:, 1], pc[:, 2], in_u_sphere=True, title="Complete input point cloud")

        sampled = sampled_pc[i][:n_sample_points]
        plot_3d_point_cloud(sampled[:, 0], sampled[:, 1], sampled[:, 2], in_u_sphere=True,
                            title="ProgressiveNet sampled points")

        recon = reconstructions[i]
        plot_3d_point_cloud(recon[:, 0], recon[:, 1], recon[:, 2], in_u_sphere=True,
                            title="Reconstruction from ProgressiveNet points")


# S-Net训练函数
def train_snet(args, pc_data_train, pc_data_val, top_out_dir):
    ae_dir = osp.join(top_out_dir, args.ae_folder)
    conf = Conf.load(osp.join(ae_dir, 'configuration'))

    # 更新配置
    conf.ae_dir = ae_dir
    conf.ae_name = 'autoencoder'
    conf.restore_ae = args.restore_ae
    conf.ae_restore_epoch = 500
    conf.fixed_ae = args.fixed_ae

    if conf.fixed_ae:
        conf.encoder_args['b_norm_decay'] = 1.0
        conf.decoder_args['b_norm_decay'] = 1.0
        conf.decoder_args['b_norm_decay_finish'] = 1.0

    conf.encoder_args['return_layer_before_symmetry'] = True

    # 采样器配置
    conf.experiment_name = 'sampler'
    conf.n_samp = [args.n_sample_points, 3]
    conf.sampler = sampler_with_convs_and_symmetry_and_fc
    conf.similarity_reg_weight = args.similarity_reg_weight
    conf.learning_rate = args.learning_rate

    train_dir = create_dir(osp.join(top_out_dir, args.train_folder))
    conf.train_dir = train_dir
    conf.save(osp.join(train_dir, 'configuration'))

    # 构建模型
    reset_tf_graph()
    ae = SNetPointNetAutoEncoder(conf.experiment_name, conf)

    # 训练采样器
    buf_size = 1
    with open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size) as fout:
        train_stats = ae.train(pc_data_train, conf, log_file=fout, held_out_data=pc_data_val)


# S-Net评估函数
def evaluate_snet(args, pc_data_test, top_out_dir):
    train_dir = osp.join(top_out_dir, args.train_folder)
    restore_epoch = 500
    conf = Conf.load(osp.join(train_dir, 'configuration'))
    conf.encoder_args['return_layer_before_symmetry'] = True

    # 恢复模型
    reset_tf_graph()
    ae = SNetPointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(train_dir, epoch=restore_epoch, verbose=True)

    n_sample_points = conf.n_samp[0]
    eval_dir = create_dir(osp.join(train_dir, "eval"))

    # 采样点云
    _, sampled_pc, sample_idx, _ = ae.get_samples(pc_data_test.point_clouds)
    np.save(osp.join(eval_dir, f"sampled_pc_test_set_{args.object_class}_{n_sample_points:04d}.npy"), sampled_pc)
    np.save(osp.join(eval_dir, f"sample_idx_test_set_{args.object_class}_{n_sample_points:04d}.npy"), sample_idx)

    # 重建点云
    reconstructions = ae.get_reconstructions_from_sampled(sampled_pc)
    np.save(osp.join(eval_dir, f"reconstructions_test_set_{args.object_class}_{n_sample_points:04d}.npy"),
            reconstructions)

    # 计算损失
    ae_loss_per_pc = ae.get_loss_ae_per_pc(pc_data_test.point_clouds, sampled_pc)
    np.save(osp.join(eval_dir, f"ae_loss_test_set_{args.object_class}_{n_sample_points:04d}.npy"), ae_loss_per_pc)

    # 保存日志
    with open(osp.join(eval_dir, f"eval_stats_test_set_{args.object_class}_{n_sample_points:04d}.txt"), "w",
              1) as log_file:
        log_file.write(f"Evaluation flags: {args}\n")
        log_file.write(f"Mean ae loss: {ae_loss_per_pc.mean():.9f}\n")

        # 计算标准化重建误差
        ref_path = osp.join(conf.ae_dir, "eval", f"ae_loss_test_set_{args.object_class}.npy")
        if osp.exists(ref_path):
            ae_loss_per_pc_ref = np.load(ref_path)
            nre_per_pc = np.divide(ae_loss_per_pc, ae_loss_per_pc_ref)
            log_file.write(f"Normalized reconstruction error: {nre_per_pc.mean():.9f}\n")

    # 可视化结果
    if args.visualize_results:
        i = 0
        pc = pc_data_test.point_clouds[i]
        plot_3d_point_cloud(pc[:, 0], pc[:, 1], pc[:, 2], in_u_sphere=True, title="Complete input point cloud")

        sampled = sampled_pc[i]
        plot_3d_point_cloud(sampled[:, 0], sampled[:, 1], sampled[:, 2], in_u_sphere=True, title="S-NET sampled points")

        recon = reconstructions[i]
        plot_3d_point_cloud(recon[:, 0], recon[:, 1], recon[:, 2], in_u_sphere=True,
                            title="Reconstruction from S-NET points")


if __name__ == '__main__':
    main()