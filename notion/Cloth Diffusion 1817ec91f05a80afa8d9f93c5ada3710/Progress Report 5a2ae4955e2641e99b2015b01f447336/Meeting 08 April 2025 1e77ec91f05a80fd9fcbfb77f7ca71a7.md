# Meeting 08 April 2025

- Performance vs # Demonstrations
    
    ![cloth_fold_performance_vs_num_dems.png](Meeting%2008%20April%202025%201e77ec91f05a80fd9fcbfb77f7ca71a7/cloth_fold_performance_vs_num_dems.png)
    

**Comment**

- DMFD Image based seems a good benchmark to evaluate sample efficiency
- BC: high variance. Why? A result like that might not be much relevant: maybe errors / bugs.. in their benchmark they put it to 0.20 for state based and to 0.0 for image based so they havent debugged that theirself: with using the same network as for DMFD, already improvement…

## **Looking Forward**

### **DMFD**

- [x]  **state**
    - [ ]  scale buffer with ratio (40, 100)
    - [ ]  fill buffer (40, 100)

### **ClothDiff**

*Discussion on different Actor usage:*

- fair benchmark
- might improve performance (mean and variance) also just with BC
- [ ]  BC with diffusion

**Privileged**

- [ ]  BC-image add privileged as done on PushT
- [ ]  Contrastive Loss

### **TrakDis**

*will it ever arrive? …*

let us do our tests, at most we will have to re-do the code we can experiment with the architecture.

- Decision Transformer
    
    DT GitHub: [https://github.com/kzl/decision-transformer](https://github.com/kzl/decision-transformer)
    
    online-DT GitHub: [https://github.com/facebookresearch/online-dt](https://github.com/facebookresearch/online-dt)