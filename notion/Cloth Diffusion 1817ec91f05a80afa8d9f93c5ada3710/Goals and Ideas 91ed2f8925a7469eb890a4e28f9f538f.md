# Goals and Ideas

# High-Level Project Questions

1. What is the high-level goal / objective of the project?
    
    Develop a method for Imitation Learning on a DOM task.
    
    The research focus is on **leveraging privileged information during training**.
    
    The objective of the research is to improve sample-efficiency and potentially performance.
    
    Constraints: The method should generalize through tasks / objects.
    
2. What is the motivation for achieving this goal? What can your method be used for if it works? 
    
    Motivation:
    
    - *Imitation learning needs lots of information, try to lower the number of samples needed for convergence.*
    - Gaining lots of real-world data on a single robot configuration is not scalable. Various ways tried to achieve this:
        - Simulation. Problem sim-to-real gap
        - Pre-training: uses diverse dataset. Very costly and not necessary in-domain.
        - …
    - Our solution: exploit privileged data. The idea is that the pipeline of gathering data does not scale efficiently with parallelization because of physical constraints, on the other hand what we can scale is the range of data that is collected through what is called in this paper as “**privileged-data**”.
        1. Privilege data:
            1. cloth graph
        2. “multimodality” and data diversity:
            1. Vision:
                1. image
                2. 3D feature lifting
                3. ? point cloud
                4. ? multiple cameras for different views on the same action
                5. ? synthetic data expansion
            2. control: Volts, end effector, dof displacement
            3. Language
                1. tradeoff between generalization and cost
        
3. How will we evaluate our method? What datasets will we evaluate on? What metrics will we use?
    
    **Baseline**
    
    - **IF** starting from a pre-existing architecture:
        
        Can use the base model / diffusion from which we start.
        
        1. Base model from the paper → using their training
        2. Base model PLUS fine-tune without privileged data
    - **IF** try to implement a new model
        1. TBD: *the architecture/s that it starts from*
    
    - [ ]  Ad hoc benchmark. mix of SIM and REAL
    - [ ]  Benchmark used inside the paper of the model / architecture
    
4. What is your high level idea of how to achieve this goal?
    
     
    
    **Dream method**
    
    Using transformers’ to match vision (3D features, …) and the privileged data so that at inference the network incorporates by design the privileged data.
    
    ! We don’t want to overfit → it should be able to generalize
    
    **Incorporating privileged data**
    
    - Key idea: task execution means that there is an implicit loss between the starting and the final space. For example, if we want to fold a T-shirt and place it back where it was, we can calculate the divergence from the cloth initial position to the final one.
    - Practically, an action (ex. “*fold T-shirt*”) connects the “*current state*” of the object to a “*desired state*”.
    - The idea here is to generate a coupling between “*language action*” and “*desired state*”. And during training is therefore able to generate actions, based solely on the “*current state*”, as it learned the connection before.
        
        To do so the exploitation of **privileged data** during training aims at defining a model that incorporates the knowledge of the desired state, or better, **the difference between the current and the desired state**, in cloth state terms, and based on that generates actions.
        
    - Therefore the key part of the research is in the **perception** part of the task, rather than the action inference, that will be inherited from previous models. This idea can take concrete form into:
        - A feature extraction from vision that, through techniques for multimodality, incorporates the privileged data.
    
5. What are some existing alternative ways to achieve this goal (based on previously developed methods), and what are the limitations of each?
    
    
    Various alternatives cover various sub-parts of the research.
    
    - Pre-training models ex [RDP-1B](https://rdt-robotics.github.io/rdt-robotics/) achieves states sample efficiency fine-tuning, not training.
    
6. What parts of your approach are most unique / different from previous approaches to this problem? (Note that all research builds on previous work so you don’t need for your approach to be completely novel, but there should be something different compared to previous work.)
    - The key of the research study is the privileged data exploitation, therefore “multimodality”
    
    The **novelty** is **how** to ****incorporate **privileged data**.
    
7. Why do you think that your approach will be better (in some way) than the alternatives?
    
    Pros:
    
    - data efficiency
    - *performance*
    
    Cons:
    
    - **Generalization** to different fields:
        
        To develop a highly versatile privileged information layer the privileged information the model has to be exposed to has to be vastly diverse. The purpose of this paper is instead narrow and vertical on DOM and particularly on the cloth manipulation task, that is a practical use case where to test the model.
        
    
8. What part of the project is most risky / most likely to fail? What are some risk mitigation strategies? (e.g. alternative approaches for this risky step; simpler problems that we can tackle that are more likely to succeed) Note that this part will likely need to get updated as we explore our method and discover which parts are most difficult (it’s often different from what we originally thought!)
    - The architecture to incorporate the privileged data
    
    Risk mitigation: start small, simple policy simple task, grow on results.
    
9. What baseline methods (from previous work) will we compare to? (Note - we may also need to compare to ablations of our method to understand the importance of each component of our method)
    - The best method on the tasks we choose to work with.
        - performance
        - training requirements
        - inference efficiency
    
10. How long do we think this project will take? (Note: the default here is 1 year; most new projects take 6 months to 1 year unless this is a direct extension to one of your previous projects)
    
    8 months
    
    - 1 month research
    - 1 month confidence with things
    - 1 month initial experiments
    - 2 weeks write initial experiments report
    - 1 month secondary experiments
    - 2 weeks write final experiments report
    - 1 month paper
    - (2 months margin, perfection does not exist)
    

It is good to keep this updated and look back at it from time to time to make sure that we are on a good research path, or to re-evaluate our direction if needed.