ROOT_DIR = '/root/dist-automl'
PYTHON_HOME = '/usr/local/anaconda3/bin/python'
MPI_HOME= '/root/software/openmpi-4.0.1'

# rdma=0, 1G Eth
MPI_PREFIX = ['%s/bin/mpirun' % MPI_HOME, '--allow-run-as-root', '--prefix', MPI_HOME, \
                 '-bind-to', 'none', '-map-by', 'slot', \
                 '-x', 'LD_LIBRARY_PATH', \
                 '-x', 'NCCL_DEBUG=INFO', \
                 '-x', 'NCCL_IB_DISABLE=1', \
                 '-x', 'NCCL_SOCKET_IFNAME=eth0', \
                 '-mca', 'pml', 'ob1', '-mca', 'btl', '^openib', \
                 '-mca', 'btl_openib_allow_ib', '0', \
                 '-mca', 'btl_tcp_if_include', 'eth0']

