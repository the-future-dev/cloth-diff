# 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations

https://3d-diffuser-actor.github.io/

![{C54F253D-10A1-48FE-BB47-BEA1F4D36771}.png](3D%20Diffuser%20Actor%20Policy%20Diffusion%20with%203D%20Scene%20R%20c17b815d14ac42b0974cbc4fe6c914cd/C54F253D-10A1-48FE-BB47-BEA1F4D36771.png)

- **Research Problem & Motivation**
    
    Combine 3D scene representations and diffusion policies for learning robot policies using IL.
    
    The limitation is the multimodality of many tasks: multiple actions can lead to optimal behavior.
    
- **List of most relevant related work cited in the paper**
    1. **Diffusion Policies**:
        - Diffusion models like Denoising Diffusion Probabilistic Models (DDPMs) are used for action prediction (e.g., [6], [31], [32]).
        - ChainedDiffuser [21]: Uses diffusion for trajectory planning but lacks tokenized 3D representations.
    2. **3D Representations**:
        - Act3D [17]: Uses 3D action maps but lacks probabilistic modeling.
        - PerAct [16]: Employs voxelized 3D workspaces but at a high computational cost.
    3. **Comparative Approaches**:
        - 2D approaches like RT-1 [49] for vision-based end-effector pose prediction.
        - 2D approaches like Perceiver-Actor [16] for vision-based end-effector pose prediction.
- **Contributions**
    
    Architecture 3D Diffuser Actor, a denoising policy transformer that takes as input:
    
    - a tokenized 3D scene representation
    - a language instruction
    - a noised end-effector’s fufture translation and rotational trajectory
    
    This approach is argued to handle multimodality in action prediction and perform spatial reasoning.
    
- **Limitations**
    1. **Camera Calibration Dependency and depth information**: may not be readily available in real-world scenarios.
    2. **Task Dynamics**: the investigation is limited to quasi-static tasks.
    3. **Inference Speed**: slower than non-diffusion methods.

# Summary of the paper

Introduction of the architecture **3D Diffuser Actor**: combining 3D scene representations with diffusion model for learning robot manipulation policies.

# Method

The method involves:

1. **Diffusion Modeling**:
    - Predicts 3D trajectories via a denoising process.
    - Integrates 3D tokens from scenes, language, and proprioception.
2. **Tokenization**:
    - Converts 3D scene points and noised trajectories into latent embeddings with positional encodings.
    - Fuses these embeddings using a 3D relative denoising transformer.
3. **Training**:
    - Supervises predictions with L1 and binary cross-entropy losses.
    - Employs keypose segmentation for trajectory sampling.
4. **Inference**:
    - Iteratively denoises trajectories, optionally using motion planners like BiRRT for keypose execution.

# List of experiments

The experiments conducted include:

- **RLBench (Simulated Benchmark):**
    - Multi-view and single-view setups.
    - Comparisons with 2D and 3D baselines.
- **CALVIN (Zero-Shot Generalization):**
    - Evaluation on unseen environments.
    - Successive task completion rates.
- **Real-World Demonstrations:**
    - Twelve diverse manipulation tasks.
    - Evaluated for generalization from sparse demonstrations.
- **Ablation Studies:**
    - Impact of 3D tokenization and relative attention.
    - Comparisons with 2D-only and absolute attention models.