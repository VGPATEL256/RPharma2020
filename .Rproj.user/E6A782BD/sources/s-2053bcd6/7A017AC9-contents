#
# Run this script to setup the keras/tensorflow environment in RStudio Cloud
#


# Install packages --------------------------------------------------------
install.packages(c(
  "keras",
  "rmarkdown",
  "rprojroot",
  "tidyverse"
))


# Define virtual environment ----------------------------------------------
root <- rprojroot::find_rstudio_root_file()
reticulate::virtualenv_create("r-reticulate")
line <- "reticulate::use_virtualenv(\"r-reticulate\")"
write(line, file.path(root, ".Rprofile"), append = TRUE)
rstudioapi::restartSession()


# Install keras / tensorflow ----------------------------------------------
keras::install_keras(method = "virtualenv", envname = "r-reticulate")
rstudioapi::restartSession()
tensorflow::tf_config()
