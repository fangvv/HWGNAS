## HWGNAS

This is the source code for our paper: **Automated Design for Hardware-aware Graph Neural Networks on Edge Devices**. A brief introduction of this work is as follows:

> Graph neural networks (GNNs) have demonstrated significant advantages in handling data from non-Euclidean domains. Given the successful application of neural architecture search (NAS) in designing convolutional and recurrent neural networks, this technique has also been extended to the design of GNNs to reduce the complexity of designing task-specific models. However, existing graph NAS approaches often overlook hardware-related metrics, which are crucial for deploying GNNs on resource-constrained edge devices. This paper proposes HWGNAS, a novel reinforcement learning-based framework for simultaneously optimizing hardware-dependent latency and hardware-independent accuracy for GNNs. The search space of HWGNAS builds upon existing graph NAS methods, with careful design choices and constraints specifically aimed at optimizing inference performance. To improve search efficiency, we propose two surrogate models to effectively predict the accuracy and latency of candidate GNN architectures, respectively. By extensive evaluations on representative edge devices, experimental results show that HWGNAS significantly outperforms the baselines in terms of model size (by up to 99.5%) and inference speed (by up to 73.4x), while maintaining competitive accuracy. Moreover, HWGNAS reduces search time by 16.5% to 75.3% compared to existing graph NAS solutions.

> 边缘设备上硬件感知图神经网络的自动化设计

This work will be published by IEEE Transactions on Network Science and Engineering. Click [here](https://doi.org/10.1109/TNSE.2025.3587645) for our paper.

## Required software

PyTorch

Check 代码说明.docx for more details.

## Citation
    @ARTICLE{11077755,
		author={Li, Xiuwen and Fang, Weiwei and Qian, Liang and Li, Haoyuan and Chen, Yanming and Xiong, Neal N.},
		journal={IEEE Transactions on Network Science and Engineering}, 
		title={Automated Design for Hardware-aware Graph Neural Networks on Edge Devices}, 
		year={2025},
		volume={},
		number={},
		pages={1-14},
		keywords={Computer architecture;Accuracy;Graph neural networks;Training;Hardware;Optimization;Computational efficiency;Search problems;Predictive models;Neural architecture search;Hardware-aware;Graph Neural Network;Neural Architecture Search;Reinforcement Learning;Edge devices},
		doi={10.1109/TNSE.2025.3587645}
	}
	
## Acknowledgements
To implement this repo, we refer to the following code:
[EGNAS](https://github.com/tjdeng/EGNAS) 

## Contact

Xiuwen Li (22120397@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
