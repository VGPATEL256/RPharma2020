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
    non-, weak- or strong binder?
  - **Q5**: Same question for the peptide `LMAFYLYEW`
  - **Q6**: Same question for the peptide `LWAFYLYEV`

<details>

<summary>When you are done thinking, click here for answers</summary>

  - **Q1**: `model %>% summary` will tell you. 280,623 at default
    architecture
  - **Q2**: Because of the high model complexicity, we are over-fitting
  - **Q3**: It randomly masks updating of some weights aiming at
    avoiding overfitting. In the script drop out is set to 0, try
    changing it to 0.1 … 0.5
  - **Q4**: `0, 7.005256e-05, 0.9999299` using the command `'LMAFYLYEV'
    %>% encode_peptide(m = bl62) %>% predict(model, .)`
  - **Q5**: `1, 3.598115e-09, 0`, likewise
  - **Q6**: `0.9999325, 6.744685e-05, 8.386781e-30`, likewise

The last question really illustrates the power here. Once you have the
model working, you no longer need to go to the laboratory to test the
peptides, meaning that you can explore the biology of the system *in
silico* at greatly reduced costs. Here, we see that the 2nd and 9th
position are very important for binding.

Explanatory machine learning is a field of great development and
importance. Once you have your model, how do we understand what the
model learned and how can we infer biology from this?

</details>
