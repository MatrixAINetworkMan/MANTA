#!/usr/bin/env bash
source net_utils/automl.conf

nworkers="${nworkers:-4}"
## shrink the kernel,depth,width
#$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
#	$params \
#	$PY train_ofa_stereo.py --task large
# shrink the kernel,depth,width
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
	$params \
	$PY train_ofa_stereo.py --task large
