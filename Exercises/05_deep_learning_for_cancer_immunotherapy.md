Exercise - Deep Learning for Cancer Immunotherapy
================

You will find an Artifical Neural Network implemented in keras capable
of predicting if a given peptide will bind, i.e. the molecular
interaction here:
[R/05\_deep\_learning\_for\_cancer\_immunotherapy.R](https://github.com/leonjessen/RPharma2019/blob/master/R/05_deep_learning_for_cancer_immunotherapy.R)
(This script is also in the directory `/R` of your RStudio session).
This exercise is based on my post on [Deep Learning for Cancer
Immunotherapy on the TensorFlow for R
blog](https://blogs.rstudio.com/tensorflow/posts/2018-01-29-dl-for-cancer-immunotherapy/).

The model is working, when all class assignment are correct, i.e. when
all of the data points appear in the diagonal in the final plot.

  - **Q1**: Why is the training performance much better than the test
    performance?
  - **Q2**: We see here a new layer type `layer_dropout`, what does this
    do and why is this useful?
  - **Q3**: What are the probabilities for the peptide `LMAFYLYEV` to be
    non-, weak- or strong binder?
  - **Q4**: Same question for the peptide `LMAFYLYEW`
  - **Q5**: Same question for the peptide `LWAFYLYEV`

<details>

<summary>When you are done thinking, click here for answers</summary>

  - **Q1**: Because of the high model complexicity, we are over-fitting
  - **Q2**: It randomly masks updating of some weights aiming af
    avoiding overfitting
  - **Q3**: `0, 7.005256e-05, 0.9999299` using the command `'LMAFYLYEV'
    %>% encode_peptide(m = bl62) %>% predict(model, .)`
  - **Q4**: `1, 3.598115e-09, 0`, likewise
  - **Q5**: `0.9999325, 6.744685e-05, 8.386781e-30`, likewise

The last question really illustrates the power here. Once you have the
model working, you no longer need to go to the laboratory to test the
peptides, meaning that you can explore the biology of the system *in
silico* at greatly reduced costs. Here, we see that the 2nd and 9th
position are very important for binding.

</details>
