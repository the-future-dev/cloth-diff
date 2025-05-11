# Future Project proposal

- Evaluation (sample-efficiency) of pretrained existing models (pi0, ..) against deformable object manipulatiom
    
    The idea is to develop a unified approach to evaluate the generalization capabilities of current state of the art foundational models for robotic manipulation. 
    
- **Benchmark** for Evaluating the **Generalization** Capabilities of Robotic Manipulation Policies
    
    *Quantify how well robot manipulation policies adapt to new objects and scenarios.*
    
    - **Project Idea:** Generate a benchmark to evaluate generalization capabilities of existing robotic manipulation models based on imitation learning.
    - **Goal:** Current methods often struggle to perform well on tasks or with objects not explicitly encountered during training. This benchmark will help define and measure generalization capabilities.

- **Explainable AI** for Robotic Manipulation Policies Learned Through Deep Learning
    
    *Understand the reasoning behind AI-driven robot manipulation decisions.*
    
    - **Project Idea:** Develop a framework that can provide meaningful explanations for the actions chosen by deep learning-based robotic manipulation policies in various situations.
    - **Goal:** Increase the transparency and interpretability of these policies, allowing understanding of why a robot performed a specific action. Enhancing explainability is crucial for building trust in robotic systems, facilitating debugging, and developing more reliable and understandable robots.
    
    https://www.anthropic.com/research/tracing-thoughts-language-model
    
- Robotic Manipulation ****VLA with **Text Output for action justification**
    
     *Enable robots to justify their actions using language.*
    
    - **Project Idea**: Utilize a Vision-Language-Action (VLA) model for robotic manipulation, with the key addition of generating textual justifications for the actions it takes. The robot will not only perform a manipulation task but will also output a natural language explanation of its reasoning behind the chosen action.
    - **Goal:** Create robotic systems that are not only capable of performing complex tasks but can also communicate their decision-making process in a human-understandable way. This will increase interpretability and therefore trust in production of realworld robotic manipulators!
    
    ![image.png](Future%20Project%20proposal%201ba7ec91f05a80a89339f3b06b71b16e/image.png)
    
    https://wayve.ai/science/lingo/
    
- **Human-in-the-Loop Robot Learning**
    
    *Fine-tune robot actions on live systems using direct human feedback.*
    
    - **Project Idea**: Implement a system that allows for direct human feedback to guide and refine the behavior of robots performing manipulation tasks in real-time using reinforcement learning.
        
        This involves a Vision-Language-Action (VLA) model executing actions, receiving human feedback (e.g., via voice commands converted to text), and then fine-tuning its policy based on this feedback.
        
    - **Goal:** Enable robots to learn complex manipulation skills more effectively and intuitively through direct interaction and guidance from humans.
    
    Pretrained VLA
    
    Action execution. + Human feedback (ex voice → text  )
    
    Finetuning on human feedback
    
    Evaluation
    
- **Improve** Deformable Object Manipulation **Generalization** (Cloth-Diff)
    
    *Improve how robots handle new (deformable?) objects testing various learning techniques.*
    
    - **Project Idea**: Investigate and implement strategies to significantly improve the generalization capabilities of diffusion-based models for manipulating deformable objects like cloth
    - **Goal:** Create more versatile and robust robotic systems capable of manipulating a wide range of deformable objects in real-world settings.

- **Distillation of VLA** for running on consumer hardware
    
    *Create efficient AI models for robot control on everyday devices.*
    
    - **Project Idea:** Explore techniques for distilling large and computationally intensive Vision-Language-Action (VLA) models into smaller, more efficient versions that can run on consumer-grade hardware like standard GPUs.
    - **Goal**: Since current state-of-the-art models often require significant computational resources, limiting their deployment in more accessible robotic platforms, make advanced VLA-based robotic control more widely available by enabling its execution on less powerful and more affordable hardware.
    
    Small Inference models able to run on a consumer GPU: distill pi0 / …
    https://www.figure.ai/news/helix
    

---

Discarded Ideas:

1. Explore possibility of adding privileged info at runtime in the project: therefore an architecture that finds the object the policy wants and extrapolates. Is the same recipe applicable for Rigids? Does it work?
2. Add “*motors*” to the existing pipeline created by paper ClothDiff: expand task from folding a T-shirt efficiently to moving
- **Developing a Machine Learning Framework for Autonomous Navigation and Manipulation (with a Legged Robot) in Unstructured Outdoor Environment**
- **Improving Robotic Picking Dexterity for Diverse Objects using Tactile Feedback and Deep Learning**