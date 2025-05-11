# Review Presentation

*From “**Goals and Ideas**”:*

1. What is the high-level goal / objective of the project?
    
    Develop a method for **Imitation Learning** on a **DOM** task.
    
    The research focus is on **leveraging privileged information during training**.
    
    The objective of the research is to improve **sample-efficiency** and potentially **performance**.
    
    Constraints: The method should **generalize** through tasks / objects.
    

---

## Components

**Imitation Learning**:

![image.png](Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f/image.png)

- agent learns to perform a task by observing and imitating an expert's behavior.

| OBSERVATION | perception of the environment through sensors |
| --- | --- |
| POLICY | learned mapping from observations to action |
| ACTION | robot’s motion commands or control signals  |
| STATE |  |

**Task: Deformable Object Manipulation**

![image.png](Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f/image%201.png)

- cloth folding
- knot tying
- food handling
- …

**Observations**

- Image
- Robot perception
- **Privileged Information**
    - SIM: “perfect” state information
    

**Policy Imitation**

- Deep Learning Models

---

## **Models Overview**

[Deep Generative Models in Robotics: A Survey on Learning from Multimodal Demonstrations](Deep%20Generative%20Models%20in%20Robotics%20A%20Survey%20on%20Lea%207486e6a1aad9449cb1a6ae671bbd1ebb.md)

| **Model Type** | **Description** | **in Robotics** | **Training** |
| --- | --- | --- | --- |
| **Sampling Models** | Generate actions directly from noise or latent variables. | - Initial sampling distributions for motion planning and optimization
-  Exploration guiding models in reinforcement learning.
- Explicit sampling models for generating grasp poses, inverse kinematic solutions, or actions in a policy.  | - Variational Autoencoders (VAEs): Reconstruction loss and KL divergence.

- Generative Adversarial Networks (GANs): Binary cross-entropy loss.

- Normalizing Flows (NFlows): Negative log-likelihood.  |
| **Energy-Based Models (EBMs)** | Output a scalar value representing the energy of an action candidate. | - Cost/reward functions for sequential decision-making problems. 

- Direct action generation.  | - Contrastive Divergence (CD): Contrastive game between negative and positive samples. 

- Supervised Learning: Occupancy loss or Signed Distance Field (SDF). 

- Neural Descriptor Fields: Euclidean distance to a target action in a learned latent space.  |
| **Diffusion Models (DMs)** | Frame data generation as an iterative denoising process. | - Generation ex: trajectory, poses, scene arrangements, etc.). 

- Modular composition.  | - Denoising score matching: Learn to denoise actions by predicting the score function.  |
| **Categorical Models** | Represent the action distribution as a discrete distribution of categories or bins.	 | - GPT-like structures for autoregressive action generation.

- Action Value Maps for grounding actions in visual observations.  | - Cross-Entropy (CE) loss: Negative log-likelihood.  

- Focal loss: For addressing class imbalance.  |
| **Mixture Density Models (MDMs)** | Output the parameters of a mixture density function representing the action distribution. ex: mean, sdt dev,. | - Visuomotor policies.  | - Negative log-likelihood.  |

## Architecture for 1:1 DOM task

**Typical Architecture to solve Object Manipulation Problems:**

1. Perception Module
2. Feature representation and understanding Module
3. Action Denoising Module

[Aloha unleashed: A simple recipe for robot dexterity](Aloha%20unleashed%20A%20simple%20recipe%20for%20robot%20dexterit%20ce06c93d05e0499b813b65af81b54c56.md)

![{FE04106F-299C-4CAB-9675-E41464E1D97E}.png](Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f/FE04106F-299C-4CAB-9675-E41464E1D97E.png)

| **Research question** | **Contributions** | **Limitations** | **Relevance** |
| --- | --- | --- | --- |
| can Imitation Learning achieve robot dexterity on complex tasks in bi-manual manipulation setting? | - it is possible!
- ALOHA-2
 | - 1:1 policy to task
- sample inefficiency | - Imitation Learning |

[3D Diffusion Policy](3D%20Diffusion%20Policy%20688485b5f70b42ec995a11bdc2f75c11.md)

![{AD25C1EA-81EE-43B6-8450-54E126348710}.png](Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f/AD25C1EA-81EE-43B6-8450-54E126348710.png)

| **Research question** | **Contributions** | **Limitations** | **Relevance** |
| --- | --- | --- | --- |
| sample efficiency in imitation learning | DP3 Architecture
- Perception: point cloud in a 3D representation
- Decision: diffusion policies | - is this the optimal 3D representation?
- generalization: NO TEXT
- long horizon | - 3D embeddings |
- [ ]  **Diff-MPC**

**With Textual Input**

- Implement as second research experiment
    
    
    | PRO | CONS | research questions |  |
    | --- | --- | --- | --- |
    | overcome 1:1 policy to task | Complexity skyrockets | how to include textual inputs? | Pre-trained LLM |
    | generalization | - cost! |  |  |
    - Models
        
        
        | **name** | **Research question** | **Contributions** | **Limitations** | **Relevance** | **Perception** | **Features embeddings** | **Action** |
        | --- | --- | --- | --- | --- | --- | --- | --- |
        | PI_0 | - web pre-train | TODO |  |  | SigLIP | Gemma 2.6B | Flow Matching denoiser |
        | 3D Diffuser actor |  |  |  |  | CLIP ResNet50 2D image encode |  |  |
        

## Privileged Learning

*From “**Goals and Ideas**”:*

1. (repeat):
    
    The research focus is on **leveraging privileged information during training**.
    
    The objective of the research is to improve **sample-efficiency** and potentially **performance**.
    
2. What is the motivation for achieving this goal? What can your method be used for if it works? 
    - MOTIVATION:
        
        Imitation learning needs lots of information, try to lower the number of samples needed.
        
        - Real-world data collection on robot configuration is not scalable without massive spending.
    - METHOD: exploit privileged data.

 

[**TraKDis: Transformer-based Knowledge Distillation**](TraKDis%20Transformer-based%20Knowledge%20Distillation%20ee9a103f30ae4dc3afabb6427a48891a.md)

![{100EEED9-AF36-479A-92EC-D43CC2447FB0}.png](Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f/100EEED9-AF36-479A-92EC-D43CC2447FB0.png)

| **Research question** | **Contributions** | **Limitations** | **Relevance** |
| --- | --- | --- | --- |
| applying RL to cloth manipulation | Knowledge Distillation Procedure:
- via Weight Initialization
- state estimation encoder | - data inefficient training
- Student-Agent performance gap | - privileged data exploitation |

**Core point of the research**

- A priori high level idea: ? *maintain the knowledge inside a Transformer-based architecture that gets activated by Image Inputs*
- Call for papers

## Task and Benchmark

- **Soft Gym**
    
    
    ![{E9211D06-A593-4D3C-BA27-E954796B34D2}.png](Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f/E9211D06-A593-4D3C-BA27-E954796B34D2.png)
    
    | **Research question** | **Contributions** | **Limitations** | **Relevance** |
    | --- | --- | --- | --- |
    | need of a benchmark for DOM manipulation | - Simulation benchmark for DOM | - High sim-to-real gap
    - no robot but “pickers” | discuss |
- Garment Lab
    
    
    [Garment Lab](Garment%20Lab%2046be8c01427944bcbdf87241306f6ef3.md)
    
    ![{1093EB69-37C7-4A97-8291-B84CFF12B031}.png](Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f/1093EB69-37C7-4A97-8291-B84CFF12B031.png)
    
    | **Research question** | **Contributions** | **Limitations** | **Relevance** |
    | --- | --- | --- | --- |
    | limitations of garment simulation and benchmark for DOM | - sim environment
    - benchmark
    - sim-to-real | - sim-to-real | Valid option! |

- Non-rigid Relative Placement through 3D Dense Diffusion
    
    ![{BDEA5A77-2259-483A-8AB9-5C2F28A7C874}.png](Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f/BDEA5A77-2259-483A-8AB9-5C2F28A7C874.png)
    
    [Non-rigid Relative Placement through 3D Dense Diffusion](Non-rigid%20Relative%20Placement%20through%203D%20Dense%20Diff%206b1be81e4a0f486d8fa7f261b9f1b221.md)
    
    | **Research question** | **Contributions** | **Limitations** | **Relevance** |
    | --- | --- | --- | --- |
    | addressing deformable object transformations | - formal definition of Cross-Displacement
    
    - method: object-centric, point-wise diffusion, leveraging 3D  | - segmentation needed
     | - if we want to dive-in this particular task! |