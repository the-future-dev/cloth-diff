# 3D Diffusion Policy

# Generalizable Visuomotor Policy Learning via Simple 3D Representations

![{AD25C1EA-81EE-43B6-8450-54E126348710}.png](Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f/AD25C1EA-81EE-43B6-8450-54E126348710.png)

- **Research Problem & Motivation**
    
    Imitation learning requires many human demonstrations: it is not sample efficient. The proposal of DP3 aims at addressing this challenge through encoding a sparsely sampled point cloud in a 3D representation and subsequently denoising random noise in a coherent action sequence through diffusion policies.
    
    The challenge of the research is:
    
    - incorporating 3D point cloud
    - maintaining fast enough inference
    - handle high-dimensionality control tasks.
- **List of most relevant related work cited in the paper**
    - **Diffusion models in robotics**:
        - DP: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
        - IBC: "Imitating Human Behaviour with Diffusion Models"
    - Visual imitation learning
        - "Perceiver-Actor: A Multi-task Transformer for Robotic Manipulation"
        - "GNFactor: Multi-task Real Robot Learning with Generalizable Neural Feature Fields"
    - Learning dexterous skills
        - "Visual Dexterity: In-hand Reorientation of Novel and Complex Object Shapes”
        - "Rotating Without Seeing: Towards In-hand Dexterity Through Touch”
- **Contributions**
    - DP3 architecture: **Perception** through encoding a sparsely sampled point cloud in a 3D representation; **Decision** subsequently denoising random noise in a coherent action sequence through diffusion policies.
    
    This model demonstrates that 3D representations (sparse point clouds) can be highly effective for visual imitation learning.
    
- **Limitations**
    - The optimal 3D representation for control is left to further investigation.
    - Additionally, DP3 has not been evaluated on tasks with long horizons.

# Summary of the paper

The paper presents 3D Diffusion Policy (DP3), an IL algorithm that incorporates 3D representation (from point clouds) into diffusion policy. DP3 is evaluated on multiple robotics tasksThe method is evaluated on a range of simulated and real-world tasks, demonstrating its effectiveness and ability to handle complex scenarios like deformable object manipulation with few demonstrations. 

# Method

DP3 architecture

1. **Perception**: perceives the environment with point cloud data and process these visual observations with an encoder to extract visual features (3D).
    1. Point Cloud Processing
    2. Compact 3D representation
        1. MLP
        2. Projection Layer
2. **Decision**: a conditional denoising diffusion policy that conditions on 3D visual features and robot poses and denoises a random Gaussian noise into actions.
    1. convolutional network-based diffusion policy
    2. DDIM as noise scheduler
    3. sample prediction

(! one policy per task)

# List of experiments

- **Simulated tasks**: 72 different tasks from 7 domains are used to assess the performance of DP3 in various scenarios.
- **Real-world tasks**: 4 challenging tasks, including deformable object manipulation and tool use, are used to evaluate DP3's effectiveness in real-world settings.
- **Ablation studies**: to analyze the impact of different choices on the performance of DP3.

![{BDE6F27D-8FB8-496E-9A00-36075C5CEAAA}.png](3D%20Diffusion%20Policy%20688485b5f70b42ec995a11bdc2f75c11/BDE6F27D-8FB8-496E-9A00-36075C5CEAAA.png)