# Future Ideas

You should regularly be thinking about ideas for future projects; write down these ideas here so you don't forget about them. These don’t have to be good ideas; coming up with project ideas is a skill and the only way to improve is to practice! Your ideas will get better over time as you gain experience in research and gain practice coming up with new ideas.

Something not-seen: incorporate RL after behavior cloning.

- BC for initial model.
- RL with reward function based on privileged information

- There is a lot of mess in this field; everybody says it found the Graal and no comparison between architectures is present. INPUT: generate a comprenensive comparison of model performances and generation of a benchmark (also ok to build on existing).

*Domanda stupida*

- *se durante il raccoglimento di dati, vengono usate camere multiple; di quanto si velocizza la convergenza?*
- +1: *privileged-data rimane lo stesso per camere differenti?*

**Trends**

| **Problem** | **Solution** | **Drawbacks** | **Open Questions** |
| --- | --- | --- | --- |
| generalization:
> task
> environment | incorporate LLMs or VLMs pre-trained on large and diverse corpora of images and texts from the web | - LARGE dataset
- cost | - best architecture for the action layer and merging: V. L. A. |
|  | Flow Matching |  |  |
| data efficiency |  |  |  |

## I need more knowledge before talking about:

- Deformable object manipulation benchmark
- Would be so cool if, with privileged data during training, and some later fine-tuning without privileged data, we could implement a huge policy on 1B parameters without the need of infinite data [**RDT-1B**](https://github.com/thu-ml/RoboticsDiffusionTransformer): so #1 let’s test if this actually helps data efficiency.

- How is the state of data augmentation in imitation learning?
    - For example: starting from a single demonstration with known task and 3D environment, how many synthetics could we generate? Ex: using a VAE, pretrained
- What is the state of online RL for robotics? In other words: after pretrain, use online RL to better adjust parameters towards end goal. (*Potentially with data augmentation in the background?*)

### A priori idea architecture

- Encoder: Diffusion policy to learn the value distribution from the data;
- Decoder: Autoregressive model to learn the action;