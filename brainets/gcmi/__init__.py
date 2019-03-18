"""Multi-dimentional Gaussian copula mutual information estimation.

| **Authors** : Robin AA. Ince
| **Original code** : https://github.com/robince/gcmi
| **Reference** :
| RAA Ince, BL Giordano, C Kayser, GA Rousselet, J Gross and PG Schyns "A statistical framework for neuroimaging data analysis based on mutual information estimated via a Gaussian copula" Human Brain Mapping (2017) 38 p. 1541-1573 doi:10.1002/hbm.23471

| **Multi-dimentional adaptation**
| **Authors** : Etienne Combrisson
| **Contact** : e.combrisson@gmail.com
"""
from .nd_gcmi import (ctransform, copnorm, nd_reshape, nd_shape_checking,  # noqa
                      nd_mi_gg, nd_gccmi_ccd)
