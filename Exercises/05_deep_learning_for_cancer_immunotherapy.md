Exercise - Deep Learning for Cancer Immunotherapy
================

You will find an Artifical Neural Network implemented in keras capable
of predicting if a given peptide will bind, i.e. the molecular
interaction here:
[R/05\_deep\_learning\_for\_cancer\_immunotherapy.R](../R/05_deep_learning_for_cancer_immunotherapy.R)
(This script is also in the directory `/R` of your RStudio session).
This exercise is based on my invited post on the RStudio AI Blog: [Deep
Learning for Cancer
Immunotherapy](https://blogs.rstudio.com/ai/posts/2018-01-29-dl-for-cancer-immunotherapy/).

The model is working, when all class assignment are correct, i.e. when
all of the data points appear in the diagonal in the final plot.

  - **Q1**: How many parameters are in the deep learning model you
    created?
  - **Q2**: Why is the training performance much better than the test
    performance?
  - **Q3**: We see here a new layer type `layer_dropout`, what does this
    do and why is this useful? Could you perhaps use this to avoid the
    issue in Q1?
  - **Q4**: What are the probabilities for the peptide `LMAFYLYEV` to be
    non-, weak- or strong binder? (Hint: Try something along the lines
    of `peptide %>% encode_peptide(encoding_matrix) %>%
    predict(ann_model, .)`)
  - **Q5**: Same question for the peptide `LMAFYLYEW`
  - **Q6**: Same question for the peptide `LWAFYLYEV`
  - **Q7**: Compare your answer with the post-hoc analysis results,
    which positions are most impacted by mutations?
  - **Q8**: Look at the sequence logo, which class of amino acid
    residues at said positions are important?

<details>

<summary>When you are done thinking, click here for answers</summary>

  - **Q1**: `model %>% summary` will tell you. 280,623 at default
    architecture
  - **Q2**: Because of the high model complexity, we are over-fitting
  - **Q3**: It randomly masks updating of some weights aiming at
    avoiding overfitting. In the script drop out is set to 0, try
    changing it to 0.1 … 0.5
  - **Q4**: `0, 7.005256e-05, 0.9999299` using the command `'LMAFYLYEV'
    %>% encode_peptide(m = bl62) %>% predict(model, .)`
  - **Q5**: `1, 3.598115e-09, 0`, likewise
  - **Q6**: `0.9999325, 6.744685e-05, 8.386781e-30`, likewise
  - **Q7/8**: The last question really illustrates the power here. Once
    you have the model working, you no longer need to go to the
    laboratory to test (all) the peptides, meaning that you can explore
    the biology of the system *in silico* at greatly reduced costs
    (*Often you would use a computational approach to limit the search
    space and then go the laboratory to validate on the predicted
    peptides*). Here, we see that the 2nd and 9th position are very
    important for binding. This is in fact a “true” finding, if you look
    at a relevant crystal structure of the peptide-MHC complex, the
    position 2 and 9 serves as “anchors” (for HLA-A\*02:01) with a clear
    preference for hydrophobic amino acid residues.

Explanatory machine learning is a field of great development and
importance. Once you have your model, how do we understand what the
model learned and how can we infer biology from this? This is where the
true ML-value is created.

*Disclaimer: If you are into the details of T-cell receptor immunology,
(neo)antigens and HLA class I haplotypes, including the impact of
self-similarity on central tolerance and conserved versus improved
binders, then yes, I did skip various details in interest of time and
workshop level. I will be happy to discuss these, so feel free to reach
out*

</details>
