Exercise - A (fictive) Case Story
================

You work as a senior data scientist in a company and you have been
handed two predictive models, a simple naive baseline model and a more
sophisticated high-complexity model. Along with the models, you have
been given a visualisation quantifying some performance metrics:

<img src="06_a_fictive_case_story_files/figure-gfm/unnamed-chunk-2-1.png" style="display: block; margin: auto;" />

*metrics:*

  - mse = mean-squared-error (low is good)
  - pcc = pearson’s correlation coefficient (high is good)
  - scc = spearman’s correlation coefficient (high is good)

It looks pretty good. A good job has been done, start with a simple
baseline model and see if you can capture more of the signal with a more
sophisticated model. Given your experience you of course want to see the
more than a few metrics summarising the model performances, so you plot
the classic machine learning measured-versus-predicted machine learning
plot:

<img src="06_a_fictive_case_story_files/figure-gfm/unnamed-chunk-3-1.png" style="display: block; margin: auto;" />

*The dashed line is the identity, i.e. when the observed values are
equal to those predicted by the model.*

Ok, evidently, the more complex model captures the signal very nicely,
whereas the more simple seem to going in the right direction, but not
really capturing the subtle differences making up this data.

Now, go to [R/06\_a\_case\_story.R](../R/06_a_case_story.R) (This script
is also in the directory `/R` of your RStudio session) and let us try to
work a bit more on this. *NB\! Try to not look at the model objects, as
this will give away the point of this particular exercise*

  - **Q1**: Discuss with your breakout room buddy: What in the world is
    going on? Same data source, same models\!?

  - **Q2**: Discuss with your breakout room buddy: What in the world is
    going on? In the “Case data new models” the performance is going
    down with higher values of alpha and in the “New data new models”,
    performance is going up with higher values of alpha\!?

  - **Q3**: Again, discuss with your break-out room partner: Interpret
    and explain what is going on here?

  - **Q4**: Again, discuss with your break-out room partner: Interpret
    and explain what is going on here?

<details>

<summary>When you are done thinking, click here for answers</summary>

The essence here is the problem of overfitting. If you peak at the
`R/89_utils_case_story.R`-file you can see, that the
`data_generator()`-function, simply returns `3*x + 2 + rnorm(n, 3)`,
i.e. a linear function with some noise added. The first models we play
around with, you can also find in said file and the simple (and right)
model is a linear regression (`lm()`), whereas the complex (and wrong)
model is a local polynomial regression `loess`, with a low
`span`-parameter (0.1) meaning a high emphasis on local fit, the result
of which is a model capturing the added noise, rather than the general
trend. This `span`-parameter is the one you end up tuning. In fact what
you end up doing is identifying, that the best value for this parameters
is something roughly above 1. If you read the helps for loess `?loess`,
you will see that for `For a > 1, all points are used`, meaning that you
are not far off from `lm`. This is also what you see in the final two
plots. The tuned complex `loess`-model is roughly overlapping your
simple `lm`-model.

Take home message: BE SUPER AWARE OF OVERFITTING\! It is potentially
detrimental to the application of your model, which can end up being
very expensive. Furthermore, in this case, it would have relatively easy
to capture early on, but here we only worked with 2 parameters, whereas
if you did the exercise with the peptides, we were working on roughly a
quarter of a million parameters. In very high dimensional parameters
space, overfitting may be a lot less obvious.

</details>
