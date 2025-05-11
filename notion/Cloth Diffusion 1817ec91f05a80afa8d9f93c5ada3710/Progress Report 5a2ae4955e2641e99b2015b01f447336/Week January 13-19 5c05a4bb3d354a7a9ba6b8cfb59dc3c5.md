# Week January >13-19

[https://docs.google.com/document/d/1yA0eH-YhV1I3QmC4xpvPktAq7hKHOLjD_MST2iJqV9Y/edit?tab=t.0](https://docs.google.com/document/d/1yA0eH-YhV1I3QmC4xpvPktAq7hKHOLjD_MST2iJqV9Y/edit?tab=t.0)

# Sample-efficient Imitation Learning for Deformable Object Manipulation Using Diffusion Models

## Project Proposal

[https://www.kth.se/social/files/670fae395785adc00376a2dd/project-proposal-sample-efficient-diffusion-models-for-manipulation.pdf](https://www.kth.se/social/files/670fae395785adc00376a2dd/project-proposal-sample-efficient-diffusion-models-for-manipulation.pdf)

Supervisors: [moletta@kth.se](mailto:moletta@kth.se) | [albertal@kth.se](mailto:albertal@kth.se)

### Goal and Objectives

[The goal of this project is to develop a sample-efficient method **for learning imitation policies for deformable object manipulation by leveraging privileged information during training. Specifically, we aim to design a diffusion-based policy that is robust to missing graph information during testing, improving computational efficiency without sacrificing performance. If the results are positive, we expect to publish this work in a top-tier robotics or machine learning conference.](https://docs.google.com/document/d/1yA0eH-YhV1I3QmC4xpvPktAq7hKHOLjD_MST2iJqV9Y/edit?tab=t.0#heading=h.5x0gr1vu2hhj)

\n

## Timeline

### Questions **General

1. *How is “Success” / “Performance” defined? Do I compare my results with someone else’s work?*
- what I understood: Take an existent architecture, re-initialize, add privileged data: does it improve training time? Performance?

Flow Matching | Poisson Flow | Diffusion models

Ex: 3D Diffusion policies train from scratch: check w and w/out privileged

ex: RDT-1B (pre-trained) fine-tune on deformable: check w and w/out privileged

ex: ARP

Obiettivo: *generalizzazione a oggetti multipli*

1. *Very general inputs from on how to make the training sample efficient? Which works to look into?*
- It seems that other papers / architectures achieved sample efficiency of 5-10 demonstrations through mainly pre-training. Do I look inside there or just branch from the previous work?
    - Testing
- *Now, for the research to be effective? Which architecture is best to research on? Diffusion policies? Pre-trained VLMs? Both?*
1. *Task: what do we want to do?*
- Model
- Dataset
    - ? existing benchmarks
    - ? data recording
    - SIM / Real
- Robot: based on task. What do we have? Recording barrier?
1. Resource limits? Computation? Cluster?
    1. High performance computer // personal cluster
    2. Cluster 80GB + parallelization
2. (Proposal) I’ll get a timeline approx definition; then new check-up.

If research focus is on sample efficiency through privileged data, I guess the task can be “reduced” to leverage pre-existing architecture by adding the embeddings of Cloth-Spatting in a pre-existent architecture. Use the pre-existent paper as baseline and compare.

### Task Plan

|  | Your task will be to: | Subtasks | Timeline |
| --- | --- | --- | --- |
| 1 | Conducting a thorough literature review to identify gaps in current research on imitation learning for rigid and deformable object manipulation. | 1.1. Analysis of works in IL
*which architecture? Diffusion | VLA | Transformers*
1.2. Analysis of previous work on incorporating privileged information into the architecture:
*ex what was used for Text-To-Image?* | _?_
1 month |
| 2 | Familiarizing with tools already available in our robotics lab such as frameworks integrating diffusion models for imitation learning and frameworks for graph tracking, such as Gaussian Splatting, both in simulation and the real world. | 2.1. Which task? One or multiple? -> dataset
2.2. Which real Robot? Which Simulator?
2.3 Thesis Definition:
• privileged information can be exploited: 3D representation, also other things?
2.4 Define: model architecture. Does a pre-trained model exist? Tradeoff fast vs big: (threshold for minimum speed = ?)
2.5 How will the dataset be structured? 
• RGB
• (Privileged info,
• action info) | 2 weeks |
| 3 | Implementing baseline models and collecting demonstration datasets in both simulated and real environments. | Dataset Collection: store on hf (*does KTH have a repo?)*
• What is enough data?
• Sim | Real
• Privileged | Non Privileged
• how split into => Training | Test | Validation | 2 month |
| 4 | Testing and evaluating the proposed hypothesis by comparing the performance of models trained with and without privileged graph information. | Training Steps (based on architecture):
• ex: connector with frozen vision model
• privileged / non privileged training & val
• … | _?_ |
| 5 | Report the results in a scientific manner | • Paper generation:
Framework explanation
Publish model weights obtained on HF | _?_ |

\n

## **Literature review**

### Survey

[A Survey on Robotic Manipulation of Deformable Objects: Recent Advances, Open Challenges and New Frontiers](https://arxiv.org/abs/2312.10419)

[A Survey on Vision-Language-Action Models for Embodied AI](https://arxiv.org/abs/2405.14093)

### Models

[ALOHA Unleashed: A Simple Recipe for Robot Dexterity](https://arxiv.org/abs/2410.13126)

Architecture: transformer-based policy with diffusion loss, based on Diffusion Policy and ACT

IN: image->CNN, proprioception->MLP

Transformer Encoder -> Transformer Decoder

Out: Noise, Noisy Actions

[3D Diffuser Actor](https://3d-diffuser-actor.github.io/)

Architecture to include 3D scene feature representation, probabilistic model.

- language encoder, attention, “3D lifting”

[3D diffusion policy](https://3d-diffusion-policy.github.io/)

small, optimized architecture. Similar performance with “Diffusion Actor”

[Cloth Splatting](https://kth-rpl.github.io/cloth-splatting/)

3D estimate of cloth from RGB and GNN

- 
    
    [](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    [Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation](https://arxiv.org/abs/2306.17817)
    

ARP: [Autoregressive Action Sequence Learning for robotic manipulation](https://arxiv.org/abs/2410.03132)

- higher performance
- higher efficiency

[FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://www.pi.website/research/fast)

Architecture: [PI_0 Pretrained VLM + action expert](https://arxiv.org/abs/2410.24164)

! Match the performance of diffusion VLAs and reducing training time by up to 5x

[Robot Diffusion Transformer 1B](https://rdt-robotics.github.io/rdt-robotics/)

- Pre-trained diffusion-based foundational model.
- *5-10 examples to fine-tune*
- Key: unified action space for cross-robot model training

[A Survey on Vision-Language-Action Models for Embodied AI](https://arxiv.org/abs/2405.14093)

[Mobile Aloha](https://mobile-aloha.github.io/)

- non deformable objects
- co-training on diverse real-world datasets
- model architectures: ACT | Diffusion Policy | VINN. Implementation: https://github.com/MarkFzp/act-plus-plus

[DeformPAM](https://deform-pam.robotflow.ai/)

- Preference data (human) to learn an implicit reward model with DPO fine-tuning
- Architecture: (ResUNet3D) -> diffusion head => Action
- Flexiv Rizon 4 and camera from the top

### Benchmarks

[SoftGym](https://github.com/Xingyu-Lin/softgym?tab=readme-ov-file)

- 10 deformable tasks
- only Sim environment
- Learning Deformable Object Manipulation from Expert Demonstrations: [https://github.com/uscresl/dmfd](https://github.com/uscresl/dmfd)

[DeformPAM](https://deform-pam.robotflow.ai/)

- Dataset: (shape nuts in T form | rope in a circle | T-shirt Unfolding)

[DaXBench](https://daxbench.github.io/)

5 different simulated tasks

[garment-tracking](https://garment-tracking.robotflow.ai/)

VR human simulation | task based: fold / make plain a cloth | hf dataset

[Flat’n’Fold](https://cvas-ug.github.io/flat-n-fold)

images are not from the robot’s perspective: there is the human in the image

- 
    
    [](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    [Open X Embodiment](https://robotics-transformer-x.github.io/)
    
- 
    
    [](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    [https://github.com/Xingyu-Lin/softgym](https://github.com/Xingyu-Lin/softgym)
    
- 
    
    [](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    [https://github.com/NVlabs/DefGraspSim](https://github.com/NVlabs/DefGraspSim)
    
- 
    
    [](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    [https://github.com/dfki-ric/deformable_gym](https://github.com/dfki-ric/deformable_gym)
    
- 
    
    [](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    [ReForm: robot learning sandbox for deformable linear objects](https://github.com/ritalaezza/gym-agx)
    
- 
    
    [](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    Le Robot
    
- 
    
    [](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    Isaac Sim [https://isaac-sim.github.io/IsaacLab/main/index.html](https://isaac-sim.github.io/IsaacLab/main/index.html)
    
- 
    
    [](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    [CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks](https://arxiv.org/abs/2112.03227)
    
- 
    
    [](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    [VLABench: Benchmark for Language-Conditioned Robotics Manipulation Tasks (Preview)](https://vlabench.github.io/)
    

### Privileged Data

[Cloth Splatting](https://kth-rpl.github.io/cloth-splatting/)

3D estimate of cloth from RGB and GNN

\n