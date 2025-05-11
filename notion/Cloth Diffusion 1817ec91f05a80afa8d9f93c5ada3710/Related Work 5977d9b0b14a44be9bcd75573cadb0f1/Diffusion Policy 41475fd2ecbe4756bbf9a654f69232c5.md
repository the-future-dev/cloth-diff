# Diffusion Policy

https://arxiv.org/pdf/2303.04137

![image.png](Diffusion%20Policy%2041475fd2ecbe4756bbf9a654f69232c5/image.png)

**Paper IDEA**

- *“conditional denoising diffusion process on robot action space”*
    
    instead of directly outputting an action, the policy infers the action-score gradient,
    conditioned on visual observations, for K denoising iterations.
    
    - Stable training

- **Denoising Diffusion Probabilistic Models**
    
    Class of generative models whose output generation is modelled as a denoising process.
    
    Diffusion Visuomotor Policy Learning
    
    - Output: robot’s action: need **temporal consistency**
        
        we commit to the action-sequence prediction produced by a diffusion model for a fixed duration before replanning
        
    - **Denoising process**: conditioned on observation
        
        DDPM approximates the conditional distribution $p(A_t |O_t)$ instead of joint distribution $p(A_t |O_t)$ in paper “*Planning with diffusion for flexible behavior synthesis*”.
        
         
        
    
    Training
    
    1. randomly draw sample from dataset; randomly select a denoising iteration;
    2. sample a random noise with appropriate variance for the iteration chosen
    3. Loss: 
        
        ![image.png](Diffusion%20Policy%2041475fd2ecbe4756bbf9a654f69232c5/image%201.png)
        
        minimizes the variational lower bound of the KL-divergence between the data distribution p(x^0) and the distribution of samples drawn from the DDPM q(x^0) using:
        
        ![image.png](Diffusion%20Policy%2041475fd2ecbe4756bbf9a654f69232c5/image%202.png)
        
- **Key Design Decision**
    1. Neural Network Architecture: 
        1. **CNN-based Diffusion Policy**
            
            Feature-wise Linear Modulation (FiLM) → for conditioning the neural network on external information ex: vision.
            
        2. **Time-series diffusion transformer**
            
            transformer architecture from minGPT [[**ref**](https://arxiv.org/abs/2206.11251)]
            
            - action with noise ⇒ are passed as input tokens for the Transformer Decoder
            - sinusoidal embedding ⇒ for diffusion iteration (k) prepended as the first token
            - Observation through MLP for observation embedding ⇒ into Decoder as input features.
            - The “gradient” $ε_θ(O_t,A_t^k, k)$ is predicted by each corresponding output token of the decoder stack.
        
        recommend starting with the CNN-based diffusion policy
        
    2. Visual Encoder
        
        Input: raw image
        
        Output: latent embedding $O_t$
        
        **ResNet-18**
        
        with modification:
        
        - global average pooling replaced with spatial softmax pooling for spatial information maintainance
        - BatchNorm to GroupNorm for stable training
        
    3. Noise Scheduler
        
        **empirically** found that the Square Cosine Schedule proposed in iDDPM Nichol and Dhariwal (2021) works best for the paper’s tasks
        
- **Properties of Diffusion Policies**
    1. Model **Multi-Modal** Action Distributions
        
        key for Behaviour Cloning, multi-modality is given by:
        
        - underlying stochastic sampling
        - stochastic initialization
    2. Position Control
        
        **Action Space**: DP for **position-control** consistently outperforms DP for velocity control
        
    3. Benefits of Action-Sequence Prediction
        - DDPM scales well with output dimensions without sacrificing the expressiveness of the model.
        - Diffusion Policy represents action in the form of a high-dimensional action sequence
            - Temporal Action consistency
                - robustness to idle actions
    4. Training Stability thanks to score function

- **Evaluation**
    1. **Tasks**
        
        
        | Robomimic | Push-T | Multimodal Block Pushing | Franka-Kitchen |
        | --- | --- | --- | --- |
    2. **Key Findings**
        
        Diffusion Policy can express:
        
        - short-horizon multimodality **(multiple ways of achieving the same immediate goal)**.
        - long-horizon multimodality **(completion of different sub-goals in inconsistent order)**.
        - better leverage position than velocity control
        - **Tradeoff in action horizon**: 8 steps
            - action horizon greater than 1 helps the policy predict consistent actions and compensate for idle portions of the demonstration
            - but too long a horizon reduces performance due to slow reaction time

….

- **Limitations and Future Work**
    1. inherits limitations from behavior cloning, such as suboptimal performance with inadequate demonstration data
        - Diffusion policy can be applied to other paradigms, such as reinforcement learning
    2. Second, diffusion policy has higher computational costs and inference latency compared to simpler methods like LSTM-GMM
    
    Future Work: new noise schedules; consistency models
    

- **Contributions**
    
    Bring diffusion policy to the field of robotics.
    
    - closed loop action sequences
    - visual conditioning
    - Transformer-Based diffusion network
    

**Push-T results from paper Diffusion Policy**

Target area coverage in format *(max performance) / (average of last 10 checkpoints)*

|  | **State Policy** | **Visual Policy** |
| --- | --- | --- |
| **Diffusion Policy-C** | 0.95 / 0.91  | 0.91 / 0.84 |
| **Diffusion Policy-T** | 0.95 / 0.79 | 0.78 / 0.66 |

Obtained using 200 demonstrations collected from a single human individual,  state-based tasks are trained for 4500 epochs, and image-based tasks for 3000 epochs.

![Diffusion Policy paper - Appendix A.4 - page 16](Diffusion%20Policy%2041475fd2ecbe4756bbf9a654f69232c5/image%203.png)

Diffusion Policy paper - Appendix A.4 - page 16

**Hyperparameters**

- CNN-based Diffusion Policy

| **Ctrl** | **To Ta Tp** | **#D-Params**  | **#V-Params** |  **Lr** | **WDecay** | **D-Iters Train** |  **D-Iters Eval** | **ImgRes** | **CropRes** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pos | 2 8 16 | 256 | 22 | 1e-4 | 1e-6 |  100 | 100 |  1x96x96 | 1x84x84 |
- Transformer-based Diffusion Policy

| **Ctrl** | **To Ta Tp** | **#D-Params**  | **#V-params** |  **Lr** | **WDecay** | **D-Iters Train** |  **D-Iters Eval** | **# Layers** | **Emb Dim** | **Attn Drp** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pos | 2 8 16 | 9 | 22 | 1e-4 | 1e-6 |  100 | 100 | 8 | 256 | 0.3 |