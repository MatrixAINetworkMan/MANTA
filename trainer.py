import os
import time
import json
import requests
import hashlib
import subprocess
import psutil
from copy import deepcopy

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
from net_utils.settings import *
import threading

LOG_ROOT = 'logs'

class JobThread(threading.Thread):
    def __init__(self, process):
        super().__init__()
        self.process = process

    def run(self):
        self.process.communicate()

def list_jobs(jtype='class'):

    logFs = os.listdir(LOG_ROOT)
    logFs = [logF for logF in logFs if jtype in logF]
    print(logFs)
    return logFs

def is_image_file(filename):

    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def query_stopped(jtype='class'):

    ps = query_running(jtype)
    jobs = list_jobs(jtype)
    jobs = [job.split("_")[0] for job in jobs]
    jobs = [job for job in jobs if not job in [p[1] for p in ps]]
    return jobs

def query_all(jtype='class'):

    rjs = query_running(jtype)
    rjs = [rj[1] for rj in rjs]
    jobs = list_jobs(jtype)
    sjs = ["_".join(job.split("_")[:-1]) for job in jobs]
    sjs = [job for job in sjs if not job in rjs]
    jobs = rjs + sjs

    j_status = ["(running)" for job in rjs]
    j_status.extend(["(stopped)" for job in sjs])
    return jobs, j_status

def stop_job(jn, jtype='class'):
    rjs = query_running(jtype)
    jns = [rj[1] for rj in rjs]
    if jn in jns:
        jid = jns.index(jn)
        pid = rjs[jid][0]
        print(pid)
        os.kill(int(pid), 9)

def query_running(jtype='class'):

    if jtype == 'class':
        kw = 'train_ofa_net.py'
    else:
        kw = 'train_ofa_stereo.py'

    user = os.getenv('USER')
    cmd = ['ps', '-ax'] 
    #print(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout = p.communicate()[0]
    ps = stdout.decode("utf-8").split('\n')
    ps = [line for line in ps if kw in line]
    ps = [line for line in ps if 'mpirun' in line]
    ps = [[line.split()[0], line.split()[-1]] for line in ps]
    return ps

def check_job_name(name=None, jtype='class'):
    if name == None:
        return 0 # invalid name
    name = name.strip()
    if name == "":
        return 0 # invalid name
    jobs, j_status = query_all(jtype)
    if name in jobs:
        jid = jobs.index(name)
        if "running" in j_status[jid]:
            return 1 # running
        else:
            return 2 # stop
    else:
        return 3 # new

def train_class(model='ofa', name='test', num_nodes=2, hosts='host1:2,host2:2', bs=16, lr=0.04):

    print(model, hosts, bs, name)

    cmd = deepcopy(MPI_PREFIX)
    cmd.append('-np')
    cmd.append(str(num_nodes*2))
    cmd.append('-H')
    cmd.append(hosts)
    cmd.append(PYTHON_HOME)
    cmd.append('-W ignore')
    cmd.append('train_ofa_net.py')
    cmd.append('--lr')
    cmd.append(str(lr))
    cmd.append('--bs')
    cmd.append(str(bs))
    cmd.append('--name')
    cmd.append(name)
    print(cmd)

    logF = 'logs/%s_class.log' % name
    with open(logF, 'a') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=f)
    print('launching cmd: ', cmd)
    print('cmd launched, waiting until it is finished...')
    JobThread(p).start()
    return p

def train_stereo(model='ofa', name='test', num_nodes=2, hosts='host1:2,host2:2', bs=1, lr=0.001):

    print(model, hosts, bs, name)

    cmd = deepcopy(MPI_PREFIX)
    cmd.append('-np')
    cmd.append(str(num_nodes*2))
    cmd.append('-H')
    cmd.append(hosts)
    cmd.append(PYTHON_HOME)
    cmd.append('-W ignore')
    cmd.append('train_ofa_stereo.py')
    cmd.append('--lr')
    cmd.append(str(lr))
    cmd.append('--bs')
    cmd.append(str(bs))
    cmd.append('--name')
    cmd.append(name)
    print(cmd)

    logF = 'logs/%s_stereo.log' % name
    with open(logF, 'a') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=f)
    print('launching cmd: ', cmd)
    print('cmd launched, waiting until it is finished...')
    JobThread(p).start()
    return p


if __name__ == '__main__':
    filename = 'data/cat.jpeg'
