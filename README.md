# Mandoline: Model Evaluation under Distribution Shift

Mayee Chen*, Karan Goel*, Nimit Sohoni*, Fait Poms, Kayvon Fatahalian, Christopher Re

---
<embed src="splash.pdf" type="application/pdf" alt="Mandoline splash figure.">


### Abstract
```
Machine learning models are often deployed in different settings than they were trained and validated on, 
posing a challenge to practitioners who wish to predict how well the deployed model will perform on a target distribution. 
If an unlabeled sample from the target distribution is available, along with a labeled sample from 
a possibly different source distribution, standard approaches such as importance weighting can be applied to estimate performance on the target. 
However, importance weighting struggles when the source and target distributions have non-overlapping support or are high-dimensional. 
Taking inspiration from fields such as epidemiology and polling, we develop Mandoline, 
a new evaluation framework that mitigates these issues. Our key insight is that practitioners may have prior knowledge about 
the ways in which the distribution shifts, which we can use to better guide the importance weighting procedure. 
Specifically, users write simple “slicing functions” — noisy, potentially correlated binary functions intended to capture 
possible axes of distribution shift to compute reweighted performance estimates. 
We further describe a density ratio estimation framework for the slices and show how its estimation error scales with slice quality and dataset size. 
Empirical validation on NLP and vision tasks shows that Mandoline can estimate performance on the target distribution up to 3x more accurately compared to standard baselines.
```

# About
If you use Mandoline in your work, please use the following BibTeX entry for citation:
```
```
