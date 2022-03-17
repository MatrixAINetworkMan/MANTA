#!/usr/bin/env bash
source net_utils/automl.conf

## stage 1, large supernet
#$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
#	$params \
#	$PY train_ofa_net.py --task large
# stage 2, shrink kernel
#$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
#	$params \
#	$PY train_ofa_net.py --task kernel
## stage 3, shrink depth
#$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
#	$params \
#	$PY train_ofa_net.py --task depth
## stage 4, shrink width
#$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
#	$params \
#	$PY train_ofa_net.py --task expand

$MPIPATH/bin/mpirun --allow-run-as-root --prefix $MPIPATH $hosts -bind-to none -map-by slot \
	$params \
	$PY train_ofa_net.py --name test


