# Aloha unleashed: A simple recipe for robot dexterity

[https://aloha-unleashed.github.io/](https://aloha-unleashed.github.io/)

![{FE04106F-299C-4CAB-9675-E41464E1D97E}.png](Review%20Presentation%202c2dee58feba4b4ba7c4f706f25c8c2f/FE04106F-299C-4CAB-9675-E41464E1D97E.png)

- **Research Problem & Motivation**
    
    Robot dexterity in complex bi-manual manipulation tasks involving deformable objects and contact-rich dynamics.
    
    The motivation is to investigate if scaling up imitation learning with large-scale data and expressive models can solve complex tasks like tying shoelaces or hanging shirts on a coat hanger.
    
- **List of most relevant related work cited in the paper**
    
    Imitation Learning:
    
    - Diffusion Policy (Chi et al., 2023) for expressive policy formulation and stable training.
    - ACT (Zhao et al., 2023) as a transformer-based architecture for action chunking.
    - Prior work on imitation learning for manipulation tasks (Brohan et al., 2023; Lynch et al., 2023).
    
    Bi-manual Manipulation:
    
    - Learning-based approaches for bimanual manipulation (Chen et al., 2022; Chitnis et al., 2020).
    - Studies on dexterous bimanual tasks like knot untying and cloth flattening.
    
    Large-scale Robot Learning:
    
    - Datasets for real-world robot learning (Walke et al., 2023; Khazatsky et al., 2024).
    - Methods for scaling up robot learning using teleoperation and autonomous data collection.
- **Contributions**
    - DEMONSTRATES: scaling up imitation learning with large-scale data and expressive models is effective for complex bi-manual manipulation tasks.
    - ALOA-2: a protocol for scalable teleoperation, enabling non-expert users to collect high-quality demonstration data.
    - Dataset (closed): Collection of over 26,000 demonstrations for five real-world tasks and over 2,000 demonstrations for three simulated tasks on the ALOHA 2 platform.
    - Training of end-to-end policies for challenging tasks like tying shoelaces and hanging shirts, achieving state-of-the-art performance.
- **Limitations**
    - Policies are trained for one task at a time, limiting generalization across multiple tasks.
    - The policy replans every 1 second, which may not be sufficient for highly reactive tasks requiring faster responses.
    - The approach requires a large number of human demonstrations per task, which can be time-consuming to collect.

# Summary of the paper

The paper presents “*ALOHA Unleashed*”, a system that leverages large-scale data collection and expressive models to **solve complex bi-manual manipulation** tasks through **imitation learning**.

By collecting over 26,000 demonstrations on the ALOHA 2 platform and training a **transformer-based diffusion policy**, the authors demonstrate the effectiveness of their approach in solving challenging tasks like tying shoelaces and hanging shirts.  

The results show that scaling up imitation learning with diverse and **large-scale data** can lead to significant advancements in robot dexterity. 

# Method

1. Data Collection
    - ALOHA 2 (bi-manual), teleoperation through human ⇒ massive amount of data.
    - Tasks include shoelace tying, gear insertion, and random kitchen stacking, each requiring precision and robustness.
2. **Policy Architecture**
    - transformer encoder-decoder model trained with Diffusion Loss.
    - Inputs: RGB images from 4 viewpoints, robot proprioception.
    - Outputs: trajectory chunks of 50 actions across 14 degrees of freedom

# List of experiments

1. **Task Performance Evaluation:**
    - Success rates were measured across five real-world tasks (Shirt, Lace, FingerReplace, GearInsert, RandomKitchen) and three simulated tasks (SingleInsertion, DoubleInsertion, MugOnPlate).
2. **Ablation Studies:**
    - **Impact of Dataset Size:** Policies were trained on different percentages of the Shirt dataset (100%, 75%, 50%, 25%) to evaluate the effect of data quantity on performance, particularly on ShirtEasy and ShirtMessy variants.
    - **Impact of Data Filtering:** The ShirtEasy dataset was filtered based on episode duration (shortest 75%, 50%, 25%) to assess the effect of demonstration quality on performance.
3. **Comparative Analysis:**
    - **Diffusion vs. L1 Regression Loss:** Performance of Diffusion Policy was compared to a baseline using L1 regression loss + Action chunking: 25% success on on ShirtMessy compared to 70% for the similar sized Diffusion Policy.
    - **Simulation Experiments:** Diffusion Policy vs ACT with L1 regression loss were evaluated on three simulated tasks (SingleInsertion, DoubleInsertion, MugOnPlate): results give lower accuracy of ACT.
4. **Behavior Analysis:**
    - Learned behaviors were analyzed across tasks, with specific observations of:
        - Recovery behaviors and retries in Shirt, GearInsert, and FingerReplace.
        - Relative gripper control in RandomKitchen.
        - Mode switching behaviors in ShirtMessy and LaceMessy.
        - Reorientation strategies in FingerReplace.
5. **Generalization Studies:**
    - Evaluated the Shirt model's ability to generalize to:
        - Unseen shirt with different size, color, and sleeve length.
        - Unseen robot in a home environment with a different background.
    - Tested generalization to out-of-distribution initial states in ShirtMessy, Lace, and RandomKitchen, revealing limitations in handling unfamiliar states (e.g., inverted objects).