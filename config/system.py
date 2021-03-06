# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

### SET THESE PATHS MANUALLY #####################################################
# Full paths are required because otherwise the code will not know where to look
# when it is executed on one of the clusters.

project_root = '/scratch_net/brossa/jdietric/PycharmProjects/mri_domain_adapt'
data_root = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/ADNI_Christian/ADNI_allfixed_allPP_robex/'
# data_root = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/ADNI_Christian/ADNI_all_allPP_robex'
# data_root = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/ADNI_Christian/ADNI_ender_selection_allPP_robex'
local_hostnames = ['brossa']

##################################################################################

log_root = os.path.join(project_root, 'log_dir')

def setup_GPU_environment():
    hostname = socket.gethostname()
    print('Running on %s' % hostname)
    if not hostname in local_hostnames:
        logging.info('Setting CUDA_VISIBLE_DEVICES variable...')
        # os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
        # This command is multi GPU compatible:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(os.environ["SGE_GPU"].split('\n'))
        logging.info('SGE_GPU is %s' % os.environ['SGE_GPU'])
        logging.info('CUDA_VISIBLE_DEVICES is %s' % os.environ['CUDA_VISIBLE_DEVICES'])