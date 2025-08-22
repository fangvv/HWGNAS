## HWGNAS

This is the source code for our paper: **Automated Design for Hardware-aware Graph Neural Networks on Edge Devices**. A brief introduction of this work is as follows:

> Graph neural networks (GNNs) have demonstrated significant advantages in handling data from non-Euclidean domains. Given the successful application of neural architecture search (NAS) in designing convolutional and recurrent neural networks, this technique has also been extended to the design of GNNs to reduce the complexity of designing task-specific models. However, existing graph NAS approaches often overlook hardware-related metrics, which are crucial for deploying GNNs on resource-constrained edge devices. This paper proposes HWGNAS, a novel reinforcement learning-based framework for simultaneously optimizing hardware-dependent latency and hardware-independent accuracy for GNNs. The search space of HWGNAS builds upon existing graph NAS methods, with careful design choices and constraints specifically aimed at optimizing inference performance. To improve search efficiency, we propose two surrogate models to effectively predict the accuracy and latency of candidate GNN architectures, respectively. By extensive evaluations on representative edge devices, experimental results show that HWGNAS significantly outperforms the baselines in terms of model size (by up to 99.5%) and inference speed (by up to 73.4x), while maintaining competitive accuracy. Moreover, HWGNAS reduces search time by 16.5% to 75.3% compared to existing graph NAS solutions.

> 图神经网络（GNN）在处理非欧几里得领域数据方面展现出显著优势。随着神经架构搜索（NAS）技术在卷积神经网络和循环神经网络设计中的成功应用，该技术也被延伸至GNN架构设计领域，以降低设计任务专用模型的复杂度。然而现有图神经网络架构搜索方法往往忽视硬件相关指标，这对于在资源受限的边缘设备上部署GNN至关重要。本文提出HWGNAS——一种基于强化学习的创新框架，可同步优化GNN的硬件相关延迟与硬件无关精度。该框架的搜索空间建立在现有图神经网络架构搜索方法基础之上，通过精心设计的选项与约束条件专门优化推理性能。为提升搜索效率，我们构建了两个代理模型，分别用于有效预测候选GNN架构的精度与延迟。通过对代表性边缘设备的大量评估，实验结果表明：HWGNAS在模型尺寸（最高达99.5%）和推理速度（最高提升73.4倍）方面显著优于基线方法，同时保持具有竞争力的准确率。此外，相较于现有图神经网络架构搜索方案，HWGNAS将搜索时间减少了16.5%至75.3%。

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
To implement this repo, we refer to the following code: [EGNAS](https://github.com/tjdeng/EGNAS).

In particular, we would like to express our gratitude to Dr. [Ao Zhou](https://scholar.google.com/citations?user=czrX_cYAAAAJ) for his guidance and advice on our research.

## Contact

Xiuwen Li (22120397@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
