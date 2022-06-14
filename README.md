# Contrastive Intrinsic Control for Unsupervised Skill Discovery

The original implementation was not very convenient to use on environments other than DMC, 
and had a lot of unused boilerplate code that made the method difficult to understand. 
So I carefully rewrote the parts I needed (unsupervised pretraining) to work with Gym Mujoco environments. 

**WARNING**: 
- only code for CIC pretraining is implemented, as I don't need finetuning in my research project
- no documentation, as this is just quick idea mockup for me

Other than that, code is working and even have some time/memory performance improvements in some places (`compute_apt_reward` computation to be specific). 

# References
```
@misc{laskin2022cic,
Author = {Michael Laskin and Hao Liu and Xue Bin Peng and Denis Yarats and Aravind Rajeswaran and Pieter Abbeel},
Title = {CIC: Contrastive Intrinsic Control for Unsupervised Skill Discovery},
Year = {2022},
Eprint = {arXiv:2202.00161},
}
```

Original [implementation](https://github.com/rll-research/cic) for DeepMind Control Suite.