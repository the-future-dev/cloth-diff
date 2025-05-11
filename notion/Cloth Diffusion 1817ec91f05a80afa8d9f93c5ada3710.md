# Cloth Diffusion

[Progress Report](Cloth%20Diffusion%201817ec91f05a80afa8d9f93c5ada3710/Progress%20Report%205a2ae4955e2641e99b2015b01f447336.md)

[TODO](Cloth%20Diffusion%201817ec91f05a80afa8d9f93c5ada3710/TODO%20e440a7f8b524419fbe49bab1b57e2b3d.md)

[Related Work](Cloth%20Diffusion%201817ec91f05a80afa8d9f93c5ada3710/Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1.md)

[Experiments](Cloth%20Diffusion%201817ec91f05a80afa8d9f93c5ada3710/Experiments%2000d8904435154b438374952c33cc1164.md)

[Goals and Ideas](Cloth%20Diffusion%201817ec91f05a80afa8d9f93c5ada3710/Goals%20and%20Ideas%2091ed2f8925a7469eb890a4e28f9f538f.md)

[Future Ideas](Cloth%20Diffusion%201817ec91f05a80afa8d9f93c5ada3710/Future%20Ideas%207ae36db1496e4a198b2b00a3e60a7a48.md)

**Project proposal:** https://www.overleaf.com/read/hpxyyzzfhsbk#d048f8

[Future Project proposal](Cloth%20Diffusion%201817ec91f05a80afa8d9f93c5ada3710/Future%20Project%20proposal%201ba7ec91f05a80a89339f3b06b71b16e.md)

**PAPER**

[Literature Review](Cloth%20Diffusion%201817ec91f05a80afa8d9f93c5ada3710/Literature%20Review%201b27ec91f05a8035b1a7c8849b1a3ee0.md)

**TODO**

- **table performance vs #dems**
    - [x]  dmfd state
    - [x]  fix gradient explosion for DMFD 200-25
    
    **Diffusion Policy**
    
    - [x]  connected dp and softgym
    - [x]  State
    - [ ]  (work in progress) Image
    
    - [ ]  **Privileged** → focus
        1. **Concatenazione + Dropout**
            
            [TrakDis] (?) Mean square error between states and features
            
            (with MLP)
            
        
        1. **Contrastive Loss between state and image encoder.**
            
            Encoder image; Encoder state ⇒ 2 latent spaces → contrastive loss ⇒ to the actor
            
            - [ ]  Rapid Motor Adaptation (paper)
            - [ ]  Geometric Contrastive Learning: prior shared projection space
        
        1. **Somma**
        
        BC-image privileged
        
    
    - [ ]  Decision Transformer → in TrakDis

- [ ]  Generate “article” on how to compile Softgym on Berzelius cluster
- Project proposals
    - [x]  Benchmark Of Generalization
    - [x]  Evaluation (sample-efficiency) of pretrained existing models (pi0, ..) against deformable object manipulatiom
    - [x]  Explainable Pretrained (PI0, …) Robotics for deployment: justification of robot actions for safe deployment
- [ ]  Formalize “Knowledge Development” approach
- Cloth Diff: put existing things and ideas inside a paper (to be evolved)
- [ ]  DMFD: can we derive the states from the feature encoder?
    - (if not, or partially) Does adding the state estimation improve?
    - (if so) Scale up the number of states to go towards meshes. Does it improve?

Diffusion: need convergence

- [ ]  simple dataset
- [ ]  codice triple check

**Report Formulation**

- Introduction:
    - motivation;
    - others research (Distillation, Rapid Motor Adaptation, …, Knowledge Distillation).
    - Gap: why is this relevant?
    - (Contribuzione / Idea) (ex:
- Related Work
- Method: Experimental Design:
    - partendo da obiettivo, come lo abbiamo affrontato, list of experimetns
- Future tests: dove vogliamo andare

Targeted venue? **International Conference on Intelligent Robots and Systems (IROS)**