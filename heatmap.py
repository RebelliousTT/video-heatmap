import matplotlib.pyplot as plt
from utils import GradCAM, show_cam_on_image, center_crop_img
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from ops.dataset import TSNDataSet
from mka_resnet12 import mkaresnet50
from mka_resnet0 import mkaresnet0
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_prec1 = 1


def main():
    load_pretrain = 'checkpoint/0Resfomer_resnet50-12-k5_a16-1234_something_RGB_segment8_e1000/ckpt.best.pth.tar'
    # load_pretrain = 'ckpt.best.pth.tar'
    global args, best_prec1
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    print('******************magic begin***************')


    model = mkaresnet50(num_classes=num_class, num_segments=args.num_segments, drop_rate=args.dropout, drop_path_rate=0.)
    bl_model = mkaresnet0(num_classes=num_class, num_segments=args.num_segments, drop_rate=args.dropout, drop_path_rate=0.)


    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    target_layers = [model.layer4]
    bl_target_layers = [bl_model.layer4]
    cudnn.benchmark = True

    normalize = GroupNormalize(input_mean, input_std)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path,
                   args.val_list,
                   num_segments=args.num_segments,
                   new_length=1,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,

                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       normalize,
                   ]),

                   first_img_transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size)]),

                   dense_sample=args.dense_sample),batch_size=1, shuffle=False,num_workers=args.workers, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).to(device)
    bl_model = torch.nn.DataParallel(bl_model, device_ids=args.gpus).to(device)

    if os.path.isfile(load_pretrain):
        print("=> loading checkpoint '{}'".format(load_pretrain))
        checkpoint = torch.load(load_pretrain)
        ##########################################
        stmodel_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        for kk, vv in stmodel_dict.items():
            if kk in model_dict:
                print(kk)
        stmodel_dict = {k: v for k, v in stmodel_dict.items() if k in model_dict}

        model_dict.update(stmodel_dict)
        model.load_state_dict(model_dict)

        print("=> loading checkpoint '{}'".format('checkpoint/0Resfomer_resnet50-0-k5_a16-1234_something_RGB_segment8_e1000/ckpt.best.pth.tar'))
        bl_checkpoint = torch.load('checkpoint/0Resfomer_resnet50-0-k5_a16-1234_something_RGB_segment8_e1000/ckpt.best.pth.tar')
        ##########################################
        bl_stmodel_dict = bl_checkpoint['state_dict']
        bl_model_dict = bl_model.state_dict()
        for kk, vv in bl_stmodel_dict.items():
            if kk in bl_model_dict:
                print(kk)
        bl_stmodel_dict = {k: v for k, v in bl_stmodel_dict.items() if k in bl_model_dict}

        bl_model_dict.update(bl_stmodel_dict)
        bl_model.load_state_dict(bl_model_dict)

    else:
        print(("=> no checkpoint found at '{}'".format(load_pretrain)))

    model.eval()
    bl_model.eval()

    with open('E:/dataset/v1/category.txt','r') as f:
        CAT=f.readlines()

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        bl_cam = GradCAM(model=bl_model, target_layers=bl_target_layers, use_cuda=True)

    for i, (CUT_img, input, target) in enumerate(val_loader):
        if i>=6:
            input_tensor = input.to(device)

            target_category = target.to(device)
            output_tensor = model(input_tensor)
            loss = criterion(output_tensor, target_category)
            prec1, prec5 = accuracy(output_tensor.data, target_category, topk=(1, 5))



            # if prec1!=0:
            log_output = ('Test: [{0}/{1}]\t'
                          'Loss {loss:.4f}\t'
                          'Prec@1 {top1:.3f}\t'
                          'Prec@5 {top5:.3f}\t'
                          'cat:{cat}'.format(i, len(val_loader), loss=loss, top1=prec1, top5=prec5, cat=CAT[target[0]]))
            print(log_output)


            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
            bl_grayscale_cam = bl_cam(input_tensor=input_tensor, target_category=target_category)

            plt.figure()
            for i in range(8):
                cami = grayscale_cam[i, :]
                bl_cami = bl_grayscale_cam[i, :]
                img=CUT_img[i].squeeze(-4).numpy()
                visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                                  cami,
                                                  use_rgb=True)
                bl_visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                                  bl_cami,
                                                  use_rgb=True)
                # plt.subplot(2, 8, i+1)
                # plt.imshow(img.astype(dtype=np.float32) / 255.)
                # plt.axis('off')
                # plt.subplot(2, 8, 8+i+1)
                # plt.imshow(visualization)
                # plt.axis('off')

                plt.subplot(8, 3, 3*i+1)
                plt.imshow(img.astype(dtype=np.float32) / 255.)
                plt.axis('off')
                plt.subplot(8, 3, 3*i+3)
                plt.imshow(visualization)
                plt.axis('off')
                plt.subplot(8, 3, 3*i+2)
                plt.imshow(bl_visualization)
                plt.axis('off')

            # plt.subplots_adjust(left=0,
            #                     bottom=0.6,
            #                     right=1,
            #                     top=1,
            #                     wspace=0.1,
            #                     hspace=0.2)
            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=0.3,
                                top=1,
                                wspace=0.2,
                                hspace=0.1)
            plt.show()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
