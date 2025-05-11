# TraKDis: Transformer-based Knowledge Distillation

![{100EEED9-AF36-479A-92EC-D43CC2447FB0}.png](Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f/100EEED9-AF36-479A-92EC-D43CC2447FB0.png)

- **Research Problem & Motivation**
    
    The paper approaches the challenge of applying RL to cloth manipulation: intricate dynamics of cloth and high dimensionality of the states makes RL impractical.
    
- **List of most relevant related work cited in the paper**
    - **SoftGym**: A benchmarking suite for deformable object manipulation, used for evaluating the proposed method.
    - **Decision Transformer (DT):** A transformer-based RL approach that reframes the sequence modeling problem as an action prediction problem, used as the basis for the proposed method.
    - **DMFD:** A state-of-the-art method for learning deformable object manipulation from expert demonstrations, used as a baseline for comparison.
    - **CURL:** An image-based RL algorithm that utilizes contrastive representation learning, also used as a baseline.
- **Contributions**
    
    **Knowledge Distillation Procedure:** Approach leveraging a state estimation encoder and pre-trained weights to transfer knowledge from a privileged agent to a vision-based agent. 
    
- **Limitations**
    - **Data efficiency in offline training**: current agent requires a large scale state and action task dataset.
    - **Performance Gap between Privileged Agent and Student Agent:** There is a performance gap between the vision-based agent and the privileged-data agent, especially in the cloth folding task.

# Summary of the paper

The paper proposes an RL method, based on the Decision Transformer, to execute knowledge distillation for learning a visual control agent. 

# Method

1. Pre-trained CNN Encoder for State Estimation: Image as input; Cloth state as output.
    
    + Image augmentation techniques in training stage. 
    
2. Knowledge Distillation via Weight Initialization: the visual control agent is initialized with the pre-trained weight of the privileged agent

# List of experiments

- **Comparison Experiments** on 3 SoftGym tasks (Cloth Fold Image, Cloth Fold Diagonal Pinned Image, Cloth Fold Diagonal Unpinned Image)
- **Ablation Study** verifies the necessity of each component of the proposed method.
- **Robustness Experiments:** The robustness of TraKDis is tested by introducing noise into the state estimation process.
- **Real World Experiments**.