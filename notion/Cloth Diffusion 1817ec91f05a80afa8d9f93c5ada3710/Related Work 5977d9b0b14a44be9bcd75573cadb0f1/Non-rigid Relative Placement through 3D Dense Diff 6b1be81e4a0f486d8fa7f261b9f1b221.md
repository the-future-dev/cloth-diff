# Non-rigid Relative Placement through 3D Dense Diffusion

![{BDEA5A77-2259-483A-8AB9-5C2F28A7C874}.png](Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f/BDEA5A77-2259-483A-8AB9-5C2F28A7C874.png)

- **Research Problem & Motivation**
    
    The paper approaches the task of *“relative placement”,* which involves predicting the position of one object in relation to another.
    
    Previous methods made progress in data-efficient learning and generalization to unseen task variations **but** there is a gap in addressing deformable object transformations.
    
- **List of most relevant related work cited in the paper**
    - DEDO.
    - Improved Denoising Diffusion Probabilistic Models
- **Contributions**
    1. Formal definition of “**cross-displacement**”: which extends the concept of relative placement to deformable objects using a dense representation.
    2. Method to predict cross-displacements through object-centric point-wise diffusion.
    3. Task benchmark for multimodal relative placement for deformable object manipulation from demonstrations, based on DEDO.

- **Limitations**
    1. **Segmentation Requirement**: the method relies on segmented action and anchor point clouds, which currently involves human effort.
    2. **Open-loop control**: limits the ability of handling complex placement tasks. 

# Summary of the paper

Introduction of TAX3D model: diffusion for non-rigid relative placement of deformable objects.

TAX3D employs a dense diffusion model to predict cross-displacements.

The method demonstrates generalization to unseen object instances, out-of-distribution scene configurations, and multimodal goals.

# Method

TAX3D: leverage diffusion models for point cloud generation.

- Train of the model to de-noise a set of per-point displacements, conditioned on the anchor and initial action point cloud (action context).

Model architecture based on Diffusion Transformer (DiT) and adapted to incorporate object-specific frames and cross-attention for scene-level reasoning.

- Cross-Displacement: directly predicts the cross-displacement.
- Cross-Point: directly encode and diffuse over the positions of the predicted goal point cloud.

# List of experiments

- **Baseline**: comparison against SOTA vasomotor policy (DP3).
- **Generalization**: ****novel anchor poses and cloth geometries.
- **Multi Goal Predictions**.
- **Real-world transfer**.