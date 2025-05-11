# Literature Review

**Model**

[Diffusion Policy](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/Diffusion%20Policy%2041475fd2ecbe4756bbf9a654f69232c5.md)

- Brings the diffusion models for behaviour cloning.
- Brings up the environment (push-T).

**Privileged Information Exploitation**

- **Learning Using Privileged Information**
    
    [LUPI: Learning Using Privileged Information Explained](Related%20Work%205977d9b0b14a44be9bcd75573cadb0f1/LUPI%20Learning%20Using%20Privileged%20Information%20Explain%201b27ec91f05a80108b87e670fb7e6a6c.md)
    
    - pros: can converge faster and sometimes better
    - cons: for a vast data size it is not significant
    
    Experiments:
    
    1. choose weights from teacher ⇒ enforce learning of the Teacher Neural Network. With a double loss: 
    2. weight example importance based on difficulty; if too difficult: weight to zero. In case of Neural Network adjust the learning rate
        - interesting because of trembling in the end of pushing: weight based on how fast we go to goal state

- **Multi-modality**

-