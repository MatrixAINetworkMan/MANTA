ROOT_DIR = '/home/comp/qiangwang/blackjack/dist-automl'
PYTHON_HOME = '/home/datasets/qiang/anaconda3/bin/python'
MPI_HOME= '/home/comp/qiangwang/software/openmpi-4.0.1'

## rdma=0, 10G Eth
#MPI_PREFIX = ['/home/esetstore/.local/openmpi-4.0.1/bin/mpirun', '--prefix', '/home/esetstore/.    local/openmpi-4.0.1', \
#                 '-bind-to', 'none', '-map-by', 'slot', \
#                 '-x', 'LD_LIBRARY_PATH', \
#                 '-x', 'NCCL_DEBUG=INFO', \
#                 '-x', 'NCCL_IB_DISABLE=1', \
#                 '-x', 'NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0', \
#                 '-mca', 'pml', 'ob1', '-mca', 'btl', '^openib', \
#                 '-mca', 'btl_openib_allow_ib', '0', \
#                 '-mca', 'btl_tcp_if_include', '192.168.0.1/24']

# rdma=1, 100G Eth
MPI_PREFIX = ['%s/bin/mpirun' % MPI_HOME, '--prefix', MPI_HOME, \
                 '-bind-to', 'none', '-map-by', 'slot', \
                 '-x', 'LD_LIBRARY_PATH', \
                 '-x', 'NCCL_DEBUG=INFO', \
                 '-x', 'NCCL_IB_DISABLE=0', \
                 '-x', 'NCCL_SOCKET_IFNAME=bond1', \
                 '-mca', 'pml', 'ob1', '-mca', 'btl', 'openib,vader,self', \
                 '-mca', 'btl_openib_allow_ib', '1', \
                 '-mca', 'btl_tcp_if_include', 'bond1']

from .settings_dev import *
