# Related Work

Task overview: [Literature Review](TODO%20e440a7f8b524419fbe49bab1b57e2b3d/Working%20Plan%2030b14584fb934426b91f7754d9373b78/Literature%20Review%20997f9388827a42af86156936c469942f.md) 

[Review Presentation](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f.md)

🚧 Work in progress:

[Template for RW](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/Template%20for%20RW%201817ec91f05a80d5bd2ed9c9a1755744.md)

[Zero Shot Offline IL via optimal tansport](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/Zero%20Shot%20Offline%20IL%20via%20optimal%20tansport%208d36ddca366745f08bcc7a6885c286a2.md)

# Multimodality

Knowledge distillation *(not smelling good)*

Multimodal learning (mask)

- Keypoints
- Depth

Multimodal contrastive representation learning

- CLIP
- Images vs Images+Privileged

[LUPI: Learning Using Privileged Information Explained](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/LUPI%20Learning%20Using%20Privileged%20Information%20Explain%201b27ec91f05a80108b87e670fb7e6a6c.md)

[Deep Generative Models in Robotics: A Survey on Learning from Multimodal Demonstrations](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/Deep%20Generative%20Models%20in%20Robotics%20A%20Survey%20on%20Lea%207486e6a1aad9449cb1a6ae671bbd1ebb.md)

# Model

- Privileged-data exploitation:
    
    [**TraKDis: Transformer-based Knowledge Distillation**](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/TraKDis%20Transformer-based%20Knowledge%20Distillation%20ee9a103f30ae4dc3afabb6427a48891a.md)
    

- Diffusion
    
    [Diffusion Policy](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/Diffusion%20Policy%2041475fd2ecbe4756bbf9a654f69232c5.md)
    
    [Aloha unleashed: A simple recipe for robot dexterity](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/Aloha%20unleashed%20A%20simple%20recipe%20for%20robot%20dexterit%20ce06c93d05e0499b813b65af81b54c56.md)
    
    [3D Diffuser Actor: Policy Diffusion with 3D Scene Representations](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/3D%20Diffuser%20Actor%20Policy%20Diffusion%20with%203D%20Scene%20R%20c17b815d14ac42b0974cbc4fe6c914cd.md)
    
    [3D Diffusion Policy](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/3D%20Diffusion%20Policy%20688485b5f70b42ec995a11bdc2f75c11.md)
    
    [Non-rigid Relative Placement through 3D Dense Diffusion](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/Non-rigid%20Relative%20Placement%20through%203D%20Dense%20Diff%206b1be81e4a0f486d8fa7f261b9f1b221.md)
    

- VLA (pre-trained LLM)
    
    [π0: A Vision-Language-Action Flow Model for General Robot Control](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/%CF%800%20A%20Vision-Language-Action%20Flow%20Model%20for%20General%202bbb252918bd41999b68edd78fc8f1a1.md)
    
    - [https://www.physicalintelligence.company/blog/openpi](https://www.physicalintelligence.company/blog/openpi)
    - [https://huggingface.co/lerobot/pi0](https://huggingface.co/lerobot/pi0)
    - https://github.com/Physical-Intelligence/openpi?tab=readme-ov-file

- [ ]  https://www.figure.ai/news/helix

- [ ]  https://github.com/moojink/openvla-oft

# Benchmark

https://github.com/AdaCompNUS/DaXBench

https://github.com/huggingface/gym-pusht/

[Garment Lab](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/Garment%20Lab%2046be8c01427944bcbdf87241306f6ef3.md)

[SoftGym](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/SoftGym%205c15766726c043bc863418823c169f69.md)

# Baseline

What methods can be considered “baselines” to our approach?

- Starting from a pre-existing architecture, we can use the base model / diffusion from which we start.
    - Base model pre-trained from the paper
    - Fine-tune without privileged data
- if we try to implement a new model ⇒ To Be Defined: based on the type.

 

# TO READ

## Related work

- [ ]  Diffusion per cloth:
- [ ]  [https://huggingface.co/blog/train-decision-transformers](https://huggingface.co/blog/train-decision-transformers)
- [ ]  [Generative Diffusion Models: a practical handbook](https://arxiv.org/abs/2412.17162)

## Paper reading: “models”

- [x]  Policy Diffusion: https://arxiv.org/abs/2303.04137
- [x]  State Space Models: https://huggingface.co/blog/lbourdois/get-on-the-ssm-train
- [ ]  Planning w Diffusion https://arxiv.org/abs/2205.09991

## Paper reading: “multimodality”

Overview: Multimodal learning

- [ ]  Semplice
- [ ]  Attention

- [ ]  https://arxiv.org/abs/1907.13098
- [ ]  https://arxiv.org/abs/2212.03858
    - [ ]  https://docs.google.com/presentation/d/1wdKOf-l5XM8RuRNj3MMywGMPLZKpofP3YZYJCStTvu8/edit?usp=sharing
- [ ]  https://arxiv.org/abs/2202.03390

- [x]  Aloha unleashed: A simple recipe for robot dexterity.
- [x]  [3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations](https://arxiv.org/pdf/2403.03954)
- [x]  3d diffuser actor: Policy diffusion with 3d scene representations.
- [x]  Cloth-splatting: 3d cloth state estimation from rgb supervision
- [x]  https://github.com/GarmentLab/GarmentLab
- [x]  [https://github.com/Xingyu-Lin/softgym](https://github.com/Xingyu-Lin/softgym)
    - [https://github.com/AdaCompNUS/DaXBench](https://github.com/AdaCompNUS/DaXBench)
    - [https://github.com/NVlabs/DefGraspSim](https://github.com/NVlabs/DefGraspSim)
- [x]  [π0: A Vision-Language-Action Flow Model for General Robot Control](https://www.physicalintelligence.company/download/pi0.pdf#page=14&zoom=100,65,566)
- [x]  [TraKDis: A Transformer-based Knowledge Distillation Approach for Visual Reinforcement Learning with Application to Cloth Manipulation](https://arxiv.org/pdf/2401.13362.pdf)
- [x]  [Deep Generative Models in Robotics: A Survey on Learning from Multimodal
Demonstrations](https://arxiv.org/pdf/2408.04380)
- [x]  [Non-rigid Relative Placement through 3D Dense Diffusion](https://arxiv.org/pdf/2410.19247)
- [ ]  [Diffusion Model Predictive Control](https://arxiv.org/pdf/2410.05364)
- [ ]  [**Garment Diffusion Models for Robot-Assisted Dressing**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10803021)
- [ ]  [ZERO-SHOT OFFLINE IMITATION LEARNING VIA OPTIMAL TRANSPORT](https://arxiv.org/pdf/2410.08751)

---

- [ ]  BiFold: Bimanual Cloth Folding with Language Guidance: [https://barbany.github.io/bifold/](https://barbany.github.io/bifold/)

- [ ]  Best RLBench paper - at the moment SAM2Act:  [https://arxiv.org/pdf/2501.18564v1](https://arxiv.org/pdf/2501.18564v1)
- [ ]  Cog ACT: [https://cogact.github.io/](https://cogact.github.io/)
- [ ]  Simpler Env: [https://github.com/simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv)
- [ ]  VQ BeT / Behavior Generation with Latent Actions: [https://arxiv.org/abs/2403.03181](https://arxiv.org/abs/2403.03181)

## Related Topics

### Learning from Human Demonstrations

- [ ]  [https://portal-cornell.github.io/motion_track_policy/](https://portal-cornell.github.io/motion_track_policy/)

### Online Fine-Tuning

- [ ]  Online RL: Improving Vision-Language-Action Model with Online Reinforcement Learning
- [ ]  Policy Decorator: [https://arxiv.org/pdf/2412.13630](https://arxiv.org/pdf/2412.13630)
- [ ]  Residual Assembly: [https://arxiv.org/pdf/2407.16677](https://arxiv.org/pdf/2407.16677)

### Contact

- [x]  [https://arxiv.org/html/2405.07237v1](https://arxiv.org/html/2405.07237v1)

**LLMs / VLMs / MLLMs Literature**

- [ ]  Mixture of experts: https://arxiv.org/abs/1701.06538
    - [ ]  Multimodal Mixture of experts: https://arxiv.org/abs/2311.09580
- [ ]  LLaVA: https://llava-vl.github.io/