# π0: A Vision-Language-Action Flow Model for General Robot Control

[https://www.physicalintelligence.company/download/pi0.pdf#page=14&zoom=100,65,566](https://www.physicalintelligence.company/download/pi0.pdf#page=14&zoom=100,65,566)

![{D08C0668-B85E-4E31-8B7F-FA5996611C0B}.png](%CF%800%20A%20Vision-Language-Action%20Flow%20Model%20for%20General%202bbb252918bd41999b68edd78fc8f1a1/D08C0668-B85E-4E31-8B7F-FA5996611C0B.png)

- **Research Problem & Motivation**
    
    The paper aims to develop a **generalist** robot policy, Pi_0, that can perform a variety of tasks in diverse physical environments.
    
    The primary motivation is expanding the current robot systems that are narrow and specialized, and trying to add versatility (typical of human intelligence).
    
    The goal is to create a “*robot foundational model*” that can learn from diverse data sources and generalize to new tasks and environments.
    
- **List of most relevant related work cited in the paper**
    1. **Vision-Language-Action (VLA) models:** *usage of pre-trained VLMs for robot control*
        - [https://robotics-transformer2.github.io/](https://robotics-transformer2.github.io/)
        - [https://openvla.github.io/](https://openvla.github.io/)
        - [https://octo-models.github.io/](https://octo-models.github.io/)
        - [https://tiny-vla.github.io/](https://tiny-vla.github.io/)
    2. **Diffusion models:** *DP for action generation (PI_0 uses flow matching variation)*
        - [diffusion-policy.cs.columbia.edu](http://diffusion-policy.cs.columbia.edu/)
        - [https://scaling-diffusion-policy.github.io/](https://scaling-diffusion-policy.github.io/)
    3. **Large-scale robot learning:** *self-supervised and autonomous data collection*
        - [https://arxiv.org/abs/1603.02199](https://arxiv.org/abs/1603.02199)
        - Open X-Embodiment: [https://robotics-transformer-x.github.io/](https://robotics-transformer-x.github.io/)
        - DROID: [https://droid-dataset.github.io/](https://droid-dataset.github.io/)
        - Bridge v2: [https://rail-berkeley.github.io/bridgedata/](https://rail-berkeley.github.io/bridgedata/)
    4. **Vision-Language-Models (VLM):**
        - Pali Gemma (Beyer et al., 2024) as the base for their model
    
- **Contributions**
    - The architecture that combines **VLM pre-training** (image, text) with a **flow matching** based action decoder (action) achieves better results than a similar architecture without the VLM being pre-trained.
    - The evaluation is based on a wide range of tasks, including laundry folding, table cleaning, and assembling boxes.
- **Limitations**
    - Limited understanding of pre-training dataset composition (types) and weights (amount).
    - Unclear measure of the transfer gained by combining highly diverse data: from different tasks and different robots.
    - Closed source architecture
    - Closed source dataset

# Summary of the paper

The paper presents π0, a policy that merges VLM **large-scale pre-training** and flow matching to achieve generality and complete dexterous manipulation tasks.

The model is trained on a large and diverse dataset from multiple robot platforms and evaluated on various tasks, and the paper presents its ability to follow **language instructions** and acquire new skills via fine-tuning.

The results show that π0 outperforms prior models, suggesting that this architecture is a step towards a **general robot control**.

# Method

Two layered MoE:

- **VLM backbone** for image and textual inputs
- **Flow matching action expert** to generate continuous actions for robotics-specific inputs

The method used by the paper tries to mimic the methods employed to generate foundational models in the LLM domain: a two stage training procedure - “pre-training” and “post-training”.

- The goal of the **pre-training** is to expose the model to a diverse range of tasks so that it can acquire broadly applicable and general physical capabilities.
    - The dataset should cover as many tasks as possible, and within each of those tasks
    should cover a diversity of behaviors.
- The goal of the **post-training** phase is to provide the model with the ability to skillfully and
fluently execute the desired downstream task.
    - The dataset should cover behaviors that are conducive to effective task execution, which should exhibit a consistent and fluent strategy.

# List of experiments

- **Out-of-box evaluation:** Evaluating the base model on tasks present in the pre-training data.

![fig 7](%CF%800%20A%20Vision-Language-Action%20Flow%20Model%20for%20General%202bbb252918bd41999b68edd78fc8f1a1/407D0445-A360-4649-9816-263B23C0D2A2.png)

fig 7

- **Language following:** Assessing the model's ability to follow language commands from humans and a high-level VLM policy. And comparison of web pre-trained (PI0) against policy initialize from scratch (PI0-small):

![Fig 9.](%CF%800%20A%20Vision-Language-Action%20Flow%20Model%20for%20General%202bbb252918bd41999b68edd78fc8f1a1/8B77FA9B-2C13-45FD-B2C0-3E664C50AB03.png)

Fig 9.

*Initialization using web-based pre-training provides a substantial improvement in “zero-shot” (pre-fine-tuning) task performance [fig9], and a slightly minor but still present better performance with fine-tuning [fig11].* 

- Learning **new dexterous tasks (fine-tuning)**:

![{FA690D56-80E8-4959-AB69-A67E3D41F65A}.png](%CF%800%20A%20Vision-Language-Action%20Flow%20Model%20for%20General%202bbb252918bd41999b68edd78fc8f1a1/FA690D56-80E8-4959-AB69-A67E3D41F65A.png)

- **Complex multi-stage tasks:** Fine-tuning the model on challenging tasks like laundry folding and table bussing.
    
    ![{432D9A2A-E460-4F94-8BC4-ABEE97ADD1AD}.png](%CF%800%20A%20Vision-Language-Action%20Flow%20Model%20for%20General%202bbb252918bd41999b68edd78fc8f1a1/432D9A2A-E460-4F94-8BC4-ABEE97ADD1AD.png)