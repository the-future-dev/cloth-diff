# LUPI: Learning Using Privileged Information Explained

https://web.stanford.edu/~kedart/files/lupi.pdf

1. **Introduction**
    
    Learning Using Privileged Information was first suggested by [1] in which they tried to capture the essence of teacher-student based learning.
    
    *We want to build a decision rule for determining labels **y** based on some features **X**, but in the training stage we are provided with “privileged information” **x*** which is not present in  the testing stage.*
    
    *In such scenario, can we utilise **X*** to improve the learning?*
    
    1. **LUPI Framework**
        
        mathematical framework
        
        Question: can the generalization performance be improved using the privileged information? Proved to be true for SVM.
        
    2. **SVM and SVM+**
        
        mathematical framework
        
    3. **Weighted SVM and margin transfer SMVs**
        
        One way in which privileged information influences learning is by differentiating easy examples from the really difficult ones. Formalized in [2]: if the weights are chosen appropriately then Weighted-SVM can always outperform SVM+.
        
        - definition of Margin Transfer SVM: margin $\rho$ determine how difficult an example is; weigh training based on $\rho$ and if an example is too difficult ($\rho$<0) then its weight is equal to 0!
2. **Experiments**
    1. **SVM+ vs SVM**
        
        SVM+ converge faster, and in some cases it converges to a better answer
        
    2. **Manual weighted SVM vs SVM+**
        
        Privileged information related to difficulty indeed does help the learning in lot of scenarios. Although, for higher data sizes, the improvement is not significant.
        
    3. **LUPI-FNN**
        
        Use intuition obtained from before to the case of Neural Networks
        
        basic idea: weights can be used to modify the learning rate per-example while applying training procedures based on gradient descent.