# Conformal Risk-Adaptive Navigation for Mobile Robot in Crowded Environments
In this paper, we propose Conformal Risk-Adaptive Model Predictive Control (CRA-MPC), the first method combining probabilistic ensemble learning with online adaptive conformal calibration to achieve real-time safety boundary adjustment in dynamic crowds. Our hierarchical framework integrates a Probabilistic Ensemble Neural Network (PENN) for long-horizon waypoint selection with an Adaptive Conformal Unit (ACU) for
short-term safety calibration, feeding uncertainty estimates into dynamic control barrier functions within risk-adaptive MPC.
**Supplemental material** is available at the provided [link](https://github.com/user-attachments/files/22711327/Supplementary_material_for_IV-1.pdf)
.

## Installation
python 3.9.0

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage 

python main.py --ctrl-type adap_cvarbf --htype dist_cone  

## Arguments:

--ctrl-type: Controller type (cbf, cvarbf, adap_cvarbf)

--htype: h-function type (dist_cone, vel, dist)

--beta: Risk parameter: fixed for cvarbf controller and adaptive for adap_cvarbf controller

## Overview of Adaptive CVaR Barrier Functions
![Overview of Adaptive CVaR Barrier Functions](/config/20obs/figures/sample.gif)



---

## Citation
