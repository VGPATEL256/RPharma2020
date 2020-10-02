# ------------------------------------------------------------------------------
# Setup session:
# Artificial Neural Networks using Keras / Tensorflow in R
# ------------------------------------------------------------------------------

# Install packages
# ------------------------------------------------------------------------------
install.packages("tidyverse")
install.packages("devtools")

# Load libraries
# ------------------------------------------------------------------------------
library("tidyverse")
library("devtools")

# Install keras
# ------------------------------------------------------------------------------
install_github("rstudio/keras")
# Would you like to install miniconda? Y
library(keras)
install_keras(tensorflow = "1.13.1")
