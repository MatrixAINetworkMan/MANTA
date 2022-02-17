# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch.nn as nn
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import horovod.torch as hvd

from ofa.utils import AverageMeter, cross_entropy_loss_with_soft_target
from ofa.utils import DistributedMetric, list_mean, subset_mean, val2list, MyRandomResizedCrop
from ofa.stereo_matching.run_manager import DistributedRunManager

__all__ = [
    'validate', 'train_one_epoch', 'train', 'load_models',
    'train_elastic_depth', 'train_elastic_expand', 'train_elastic_width_mult',
]


def validate(run_manager, epoch=0, is_test=False, image_size_list=None,
             ks_list=None, expand_ratio_list=None, depth_list=None, scale_list=None, width_mult_list=None, additional_setting=None):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    dynamic_net.eval()

    if image_size_list is None:
        image_size_list = val2list(run_manager.run_config.data_provider.image_size, 1)
    if ks_list is None:
        ks_list = dynamic_net.ks_list
    if expand_ratio_list is None:
        expand_ratio_list = dynamic_net.expand_ratio_list
    if depth_list is None:
        depth_list = dynamic_net.depth_list
    if scale_list is None:
        scale_list = dynamic_net.scale_list
    if width_mult_list is None:
        if 'width_mult_list' in dynamic_net.__dict__:
            width_mult_list = list(range(len(dynamic_net.width_mult_list)))
        else:
            width_mult_list = [0]

    subnet_settings = []

    #ds = [4, 2, 2]
    #es = [8, 8, 2]
    #ks = [7, 7, 3]
    #ss = [4, 2, 2]
    #img_size = 224
    #w = 0
    #for d,e,k,s in zip(ds,es,ks,ss):
    #    subnet_settings.append([{
    #        'image_size': img_size,
    #        'd': d,
    #        'e': e,
    #        'ks': k,
    #        's': s,
    #        'w': w,
    #    }, 'R%s-D%s-E%s-K%s-S%s-W%s' % (img_size, d, e, k, s, w)])
    for d in depth_list:
        for e in expand_ratio_list:
            for k in ks_list:
                for s in scale_list:
                    for w in width_mult_list:
                        for img_size in image_size_list:
                            subnet_settings.append([{
                                'image_size': img_size,
                                'd': d,
                                'e': e,
                                'ks': k,
                                's': s,
                                'w': w,
                            }, 'R%s-D%s-E%s-K%s-S%s-W%s' % (img_size, d, e, k, s, w)])
    if additional_setting is not None:
        subnet_settings += additional_setting

    losses_of_subnets, epe_of_subnets, d1_of_subnets, thres1_of_subnets, thres2_of_subnets, thres3_of_subnets = [], [], [], [], [], []

    valid_log = ''
    #print(subnet_settings)
    for setting, name in subnet_settings:
        run_manager.write_log('-' * 30 + ' Validate %s ' % name + '-' * 30, 'train', should_print=False)
        #run_manager.run_config.data_provider.assign_active_img_size(setting.pop('image_size'))
        dynamic_net.set_active_subnet(**setting)
        run_manager.write_log(dynamic_net.module_str, 'train', should_print=False)

        run_manager.reset_running_statistics(dynamic_net)
        loss, (epe, d1, thres1, thres2, thres3) = run_manager.validate(epoch=epoch, is_test=is_test, run_str=name, net=dynamic_net)
        losses_of_subnets.append(loss)
        epe_of_subnets.append(epe)
        d1_of_subnets.append(d1)
        thres1_of_subnets.append(thres1)
        thres2_of_subnets.append(thres2)
        thres3_of_subnets.append(thres3)
        valid_log += '%s (%.3f), ' % (name, epe)

    return list_mean(losses_of_subnets), list_mean(epe_of_subnets), list_mean(d1_of_subnets), valid_log


def train_one_epoch(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.network
    distributed = isinstance(run_manager, DistributedRunManager)

    # switch to train mode
    dynamic_net.train()
    if distributed:
        run_manager.run_config.train_loader.sampler.set_epoch(epoch)
    MyRandomResizedCrop.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()
    losses = DistributedMetric('train_loss') if distributed else AverageMeter()
    metric_dict = run_manager.get_metric_dict()

    with tqdm(total=nBatch,
              desc='Train Epoch #{}'.format(epoch + 1),
              disable=distributed and not run_manager.is_root) as t:
        end = time.time()
        for i, sample in enumerate(run_manager.run_config.train_loader):

            MyRandomResizedCrop.BATCH = i
            data_time.update(time.time() - end)
            if epoch < warmup_epochs:
                new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                    run_manager.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                )
            else:
                new_lr = run_manager.run_config.adjust_learning_rate(
                    run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                )

            left = sample['left'].cuda()  # [B, 3, H, W]
            right = sample['right'].cuda()
            gt_disp = sample['disp'].cuda()  # [B, H, W]

            dynamic_net.zero_grad()

            loss_of_subnets = []
            # compute output
            subnet_str = ''
            loss_type = ''
            subnet_seed = 0
            if args.task != 'large':
                for _ in range(args.dynamic_batch_size):
                    # set random seed before sampling
                    subnet_seed = int('%d%.3d%.3d' % (epoch * nBatch + i, _, 0))
                    random.seed(subnet_seed)
                    subnet_settings = dynamic_net.sample_active_subnet()
                    subnet_str += '%d: ' % _ + ','.join(['%s_%s' % (
                        key, '%.1f' % subset_mean(val, 0) if isinstance(val, list) else val
                    ) for key, val in subnet_settings.items()]) + ' || '

                    pred_disp_pyramid = run_manager.net(left, right)  # list of H/12, H/6, H/3, H/2, H
                    pred_disp = pred_disp_pyramid[-1]
                    mask = (gt_disp > 0) & (gt_disp < 192)

                    # measure accuracy and record loss
                    loss = run_manager.train_criterion(pred_disp_pyramid, gt_disp, mask)
                    loss_of_subnets.append(loss)
                    run_manager.update_metric(metric_dict, pred_disp, gt_disp, mask)

                    if not loss is None:
                        loss.backward()
            else:
                pred_disp_pyramid = run_manager.net(left, right)  # list of H/12, H/6, H/3, H/2, H
                pred_disp = pred_disp_pyramid[-1]
                mask = (gt_disp > 0) & (gt_disp < 192)

                total_loss = run_manager.train_criterion(pred_disp_pyramid, gt_disp, mask)
                loss_of_subnets.append(total_loss)
                run_manager.update_metric(metric_dict, pred_disp, gt_disp, mask)
                if not total_loss is None:
                    total_loss.backward()

            run_manager.optimizer.step()

            loss_of_subnets = [0 if loss is None else loss for loss in loss_of_subnets]
            losses.update(list_mean(loss_of_subnets), left.size(0))

            t.set_postfix({
                'loss': losses.avg.item(),
                **run_manager.get_metric_vals(metric_dict, return_dict=True),
                'R': left.size(2),
                'lr': new_lr,
                'loss_type': loss_type,
                'seed': str(subnet_seed),
                'str': subnet_str,
                'data_time': data_time.avg,
            })
            t.update(1)
            end = time.time()

            if i % 10 == 0:
                wdlog = {}
                wdlog["train_loss_iters"] = losses.avg.item()
                if hvd.rank() == 0:
                    wandb.log(wdlog)
    return losses.avg.item(), run_manager.get_metric_vals(metric_dict)


def train(run_manager, args, validate_func=None):
    distributed = isinstance(run_manager, DistributedRunManager)
    if validate_func is None:
        validate_func = validate

    ## validate the pre-loaded model
    #val_loss, val_epe, val_d1, _val_log = validate_func(run_manager, epoch=-1, is_test=False)
    ## best_acc
    #is_best = val_epe < run_manager.best_epe
    #run_manager.best_epe = min(run_manager.best_epe, val_epe)
    #if not distributed or run_manager.is_root:
    #    val_log = 'Valid [{0}/{1}] loss={2:.3f}, epe={3:.3f} ({4:.3f})'. \
    #        format(epoch + 1 - args.warmup_epochs, run_manager.run_config.n_epochs, val_loss, val_epe,
    #               run_manager.best_epe)
    #    val_log += ', Train epe {epe:.3f}, Train loss {loss:.3f}\t'.format(epe=train_epe, loss=train_loss)
    #    val_log += _val_log
    #    run_manager.write_log(val_log, 'valid', should_print=False)

    for epoch in range(run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs):
        train_loss, (train_epe, train_d1, thres1, thres2, thres3) = train_one_epoch(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr)

        if (epoch + 1) % args.validation_frequency == 0:
            val_loss, val_epe, val_d1, _val_log = validate_func(run_manager, epoch=epoch, is_test=False)
            # best_acc
            is_best = val_epe < run_manager.best_epe
            run_manager.best_epe = min(run_manager.best_epe, val_epe)
            if not distributed or run_manager.is_root:
                val_log = 'Valid [{0}/{1}] loss={2:.3f}, epe={3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - args.warmup_epochs, run_manager.run_config.n_epochs, val_loss, val_epe,
                           run_manager.best_epe)
                val_log += ', Train epe {epe:.3f}, Train loss {loss:.3f}\t'.format(epe=train_epe, loss=train_loss)
                val_log += _val_log
                run_manager.write_log(val_log, 'valid', should_print=False)

                run_manager.save_model({
                    'epoch': epoch,
                    'best_epe': run_manager.best_epe,
                    'optimizer': run_manager.optimizer.state_dict(),
                    'state_dict': run_manager.network.state_dict(),
                }, is_best=is_best, model_name=args.name)


def load_models(run_manager, dynamic_net, model_path=None):
    # specify init path
    init = torch.load(model_path, map_location='cpu')['state_dict']
    dynamic_net.load_state_dict(init)
    run_manager.write_log('Loaded init from %s' % model_path, 'valid')


def train_elastic_depth(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    depth_stage_list = dynamic_net.depth_list.copy()
    depth_stage_list.sort(reverse=True)
    n_stages = len(depth_stage_list) - 1
    current_stage = n_stages - 1

    # load pretrained models
    if run_manager.start_epoch == 0:
        validate_func_dict['depth_list'] = sorted(dynamic_net.depth_list)

        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        # validate after loading weights
        #run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
        #                      validate(run_manager, is_test=True, **validate_func_dict), 'valid')

    run_manager.write_log(
        '-' * 30 + 'Supporting Elastic Depth: %s -> %s' %
        (depth_stage_list[:current_stage + 1], depth_stage_list[:current_stage + 2]) + '-' * 30, 'valid'
    )
    # add depth list constraints
    if len(set(dynamic_net.ks_list)) == 1 and len(set(dynamic_net.expand_ratio_list)) == 1:
        validate_func_dict['depth_list'] = depth_stage_list
    else:
        validate_func_dict['depth_list'] = sorted({min(depth_stage_list), max(depth_stage_list)})

    # train
    train_func(
        run_manager, args,
        lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict)
    )


def train_elastic_expand(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    expand_stage_list = dynamic_net.expand_ratio_list.copy()
    expand_stage_list.sort(reverse=True)
    n_stages = len(expand_stage_list) - 1
    current_stage = n_stages - 1

    # load pretrained models
    if run_manager.start_epoch == 0:
        validate_func_dict['expand_ratio_list'] = sorted(dynamic_net.expand_ratio_list)

        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        dynamic_net.re_organize_middle_weights(expand_ratio_stage=current_stage)
        #run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
        #                      validate(run_manager, is_test=True, **validate_func_dict), 'valid')

    run_manager.write_log(
        '-' * 30 + 'Supporting Elastic Expand Ratio: %s -> %s' %
        (expand_stage_list[:current_stage + 1], expand_stage_list[:current_stage + 2]) + '-' * 30, 'valid'
    )
    if len(set(dynamic_net.ks_list)) == 1 and len(set(dynamic_net.depth_list)) == 1:
        validate_func_dict['expand_ratio_list'] = expand_stage_list
    else:
        validate_func_dict['expand_ratio_list'] = sorted({min(expand_stage_list), max(expand_stage_list)})

    # train
    train_func(
        run_manager, args,
        lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict)
    )

def train_elastic_scale(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    scale_stage_list = dynamic_net.scale_list.copy()
    scale_stage_list.sort(reverse=True)
    n_stages = len(scale_stage_list) - 1
    current_stage = n_stages - 1

    # load pretrained models
    if run_manager.start_epoch == 0:
        validate_func_dict['scale_list'] = sorted(dynamic_net.scale_list)

        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        #dynamic_net.re_organize_middle_weights(expand_ratio_stage=current_stage)
        #run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
        #                      validate(run_manager, is_test=True, **validate_func_dict), 'valid')

    run_manager.write_log(
        '-' * 30 + 'Supporting Elastic Scale: %s -> %s' %
        (scale_stage_list[:current_stage + 1], scale_stage_list[:current_stage + 2]) + '-' * 30, 'valid'
    )
    if len(set(dynamic_net.ks_list)) == 1 and len(set(dynamic_net.depth_list)) == 1 and len(set(dynamic_net.expand_ratio_list)) == 1:
        validate_func_dict['scale_list'] = scale_stage_list
    else:
        validate_func_dict['scale_list'] = sorted({min(scale_stage_list), max(scale_stage_list)})

    # train
    train_func(
        run_manager, args,
        lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict)
    )


def train_elastic_width_mult(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    width_stage_list = dynamic_net.width_mult_list.copy()
    width_stage_list.sort(reverse=True)
    n_stages = len(width_stage_list) - 1
    current_stage = n_stages - 1

    if run_manager.start_epoch == 0:
        load_models(run_manager, dynamic_net, model_path=args.ofa_checkpoint_path)
        if current_stage == 0:
            dynamic_net.re_organize_middle_weights(expand_ratio_stage=len(dynamic_net.expand_ratio_list) - 1)
            run_manager.write_log('reorganize_middle_weights (expand_ratio_stage=%d)'
                                  % (len(dynamic_net.expand_ratio_list) - 1), 'valid')
            try:
                dynamic_net.re_organize_outer_weights()
                run_manager.write_log('reorganize_outer_weights', 'valid')
            except Exception:
                pass
        run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
                              validate(run_manager, is_test=True, **validate_func_dict), 'valid')

    run_manager.write_log(
        '-' * 30 + 'Supporting Elastic Width Mult: %s -> %s' %
        (width_stage_list[:current_stage + 1], width_stage_list[:current_stage + 2]) + '-' * 30, 'valid'
    )
    validate_func_dict['width_mult_list'] = sorted({0, len(width_stage_list) - 1})


    # train
    train_func(
        run_manager, args,
        lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict)
    )
