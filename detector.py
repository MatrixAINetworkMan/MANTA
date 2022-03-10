import os
import time
import json
import requests
import hashlib

import torch
from PIL import Image
from torchvision import transforms, models
import argparse
import time
from ofa.model_zoo import ofa_net
import numpy as np
import skimage
from utils import set_running_statistics
from ofa.stereo_matching.elastic_nn.networks.ofa_aanet import OFAAANet
from translator import translate, get_word_result
def is_image_file(filename):

    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def images_to_tensors(folder):

    imagelist = []
    for parent, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if is_image_file(filename):
                imagelist.append(os.path.join(parent, filename))

    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
                        ])

    img_tensors = torch.zeros(1, 3, 224, 224)
    for imgname in imagelist:
        img = Image.open(imgname).convert('RGB')
        batch_t = torch.unsqueeze(transform(img), 0)
        img_tensors = torch.cat((img_tensors, batch_t))
    img_tensors = img_tensors[1:, :, :, :]
    return img_tensors

def detect_class(filename, model=None, ckpt=None):

    print(model, ckpt)
    model = model.lower()

    if 'resnet' in model:
        net = models.resnet101(pretrained=True)
    elif 'ofa' in model:
        net = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=False)
        model_file = ckpt
        init = torch.load(model_file, map_location='cpu')
        model_dict = init['state_dict']
        net.load_state_dict(model_dict)
    
        net.set_active_subnet(ks=7, e=6, d=4)
        #net.sample_active_subnet()
        #subnet = net.get_active_subnet(preserve_weight=True)
        #img_tensors = images_to_tensors('data/class_samples')
        #set_running_statistics(net, img_tensors)
    
    net.eval()

    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                        )])
    img = Image.open(filename).convert('RGB')
    batch_t = torch.unsqueeze(transform(img), 0)
    
    output = net(batch_t)
    
    print(output.size())
    
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    
    prob = torch.nn.functional.softmax(output, dim=1)[0] * 100
    _, indices = torch.sort(output, descending=True)

    result_classes = [classes[idx].split()[-1].replace('_', ' ') for idx in indices[0][:5]]
    print(result_classes)
    #result_classes = [get_word_result(translate(rc)) for rc in result_classes]
    #print(result_classes)
    result_probs = [prob[idx].item() for idx in indices[0][:5]]
    return [(rc, rp) for rc, rp in zip(result_classes, result_probs)]
    #return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

def detect_stereo(img_left, img_right, model=None, ckpt=None):

    ofa_network = OFAAANet(ks_list=[3,5,7], expand_ratio_list=[2,4,6,8], depth_list=[2,3,4], scale_list=[2,3,4])
    
    model_file = ckpt
    init = torch.load(model_file, map_location='cpu')
    model_dict = init['state_dict']
    ofa_network.load_state_dict(model_dict)
    
    d = 4
    e = 8
    ks = 7
    s = 4
    ofa_network.set_active_subnet(ks=ks, d=d, e=e, s=s)
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    #save_path = "checkpoints/aanet_D%d_E%d_K%d_S%d" % (d, e, ks, s)
    #torch.save(subnet.state_dict(), save_path)
    #subnet.load_bn_stats('checkpoints/test_bn_stats.npy')
    
    net = subnet
    net.eval()
    
    transform = transforms.Compose([
                        transforms.Resize([576, 960]),
                        transforms.ToTensor(),
                        transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
                        ])

    img_left = torch.unsqueeze(transform(img_left), 0)
    img_right = torch.unsqueeze(transform(img_right), 0)

    with torch.no_grad():
        disp_left = net(img_left, img_right)[-1]
    print(disp_left.size())
    #skimage.io.imsave('test_disp.png', (disp_left[0, :, :].numpy()*256).astype('uint16'))
    #disp_left = (disp_left[0, :, :].numpy()*256).astype('uint16')
    disp_left[disp_left > 192] = 192
    disp_left /= 192.0
    disp_img = transforms.ToPILImage()(disp_left)#.convert("RGB")
    return disp_img

if __name__ == '__main__':
    filename = 'data/cat.jpeg'
    detect(filename)
