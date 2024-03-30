## Playing around with the LASER technique
This is a simple repository to play around with the LASER technique as defined in 

```
@misc{sharma2023truth,
      title={The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction}, 
      author={Pratyusha Sharma and Jordan T. Ash and Dipendra Misra},
      year={2023},
      eprint={2312.13558},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

available at [arXiv link](https://arxiv.org/abs/2312.13558).

The LASER technique is a simple technique to reduce the rank of the layers in a transformer model through single value decomposition.

The repository contains a simple implementation of the LASER technique in PyTorch. 
The implementation is tested on the human eval set.

