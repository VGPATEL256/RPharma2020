# Scaled down version of this blog post on the TensorFlow for R Blog at RStudio
# https://blogs.rstudio.com/tensorflow/posts/2018-01-29-dl-for-cancer-immunotherapy/

# Clear workspace
# ------------------------------------------------------------------------------
rm(list = ls())

# Load libraries
# ------------------------------------------------------------------------------
library('tidyverse')
library('keras')

# Define functions
# ------------------------------------------------------------------------------
source(file = 'R/99_proj_func.R')

# Load data
# ------------------------------------------------------------------------------
file = 'data/ran_peps_netMHCpan40_predicted_A0201_reduced_cleaned_balanced.tsv'
pep_dat = read_tsv(file = file)

# About the data
# ------------------------------------------------------------------------------

# The data set consists of a total of 23,760 data points with 3 balanced classes
# each with 7,920 data points. Epitope identification is paramount in cancer
# immunotherapy. A peptide may be an epitope, if it is a strong-binder (SB),
# less likely if it is a weak-binder (WB) and not if it is a non-binder (NB)
#
# The data was created in silico using the NetMHCpan 4.0 Server.
# NetMHCpan is the state-of-the-art binding predictor and available at:
# http://www.cbs.dtu.dk/services/NetMHCpan/
#
# Aim: Create a predictive model capable of predicting if a given peptide is
#      non-binder, weak binder or SB = strong binder. I.e. a 3-class classifier
#
# Variables:
#
# - peptide
#       The amino acid sequence of the 9-mer peptide
# - label_chr
#       The class, NB = non-binder, WB = weak binder, SB = strong binder
# - label_num
#       Numeric class label 0 = NB, 1 = WB, 2 = SB
# - data_type
#       Data partitioned into a test (~10%) and training set (~90%)

# View the data
pep_dat %>% print

# View class distribution
pep_dat %>% count(label_chr, data_type) %>% print

# Prepare data
# ------------------------------------------------------------------------------

# We will use the BLOSUM62 matrix to convert the peptides to a numeric 
# representation
# Reference: https://en.wikipedia.org/wiki/BLOSUM
# Data source: https://www.ncbi.nlm.nih.gov/Class/FieldGuide/BLOSUM62.txt
bl62 = read.table(file = 'data/BLOSUM62.txt')

# Define training and test feature matrices
X_train = pep_dat %>%
  filter(data_type == 'train') %>%
  pull(peptide) %>% 
  encode_peptide(m = bl62)
X_test = pep_dat %>%
  filter(data_type == 'test') %>%
  pull(peptide) %>% 
  encode_peptide(m = bl62)

# Define known target classes for trainng and test data
y_train = pep_dat %>%
  filter(data_type == 'train') %>%
  pull(label_num) %>% 
  to_categorical
y_test = pep_dat %>%
  filter(data_type == 'test') %>%
  pull(label_num) %>% 
  to_categorical

# Define ANN model
# ------------------------------------------------------------------------------

# Set hyperparameters
n_hidden_1 = 180
h1_activate = 'relu'
drop_out_1 = 0.4
n_hidden_2 = 90
h2_activate = 'relu'
drop_out_2 = 0.3
n_hidden_3 = 45
h3_activate = 'relu'
drop_out_3 = 0.2
n_hidden_4 = 20
h4_activate = 'relu'
drop_out_4 = 0.1
n_output   = 3
o_ativate  = 'softmax'
n_epochs = 15
batch_size = 50
loss_func = 'categorical_crossentropy'
learn_rate = 0.001

# Set architecture
model = keras_model_sequential() %>% 
  layer_dense(units = n_hidden_1, activation = h1_activate, input_shape = 180) %>% 
  layer_dropout(rate = drop_out_1) %>% 
  layer_dense(units = n_hidden_2, activation = h2_activate) %>%
  layer_dropout(rate = drop_out_2) %>%
  layer_dense(units = n_hidden_3, activation = h3_activate) %>%
  layer_dropout(rate = drop_out_3) %>%
  layer_dense(units = n_hidden_4, activation = h4_activate) %>%
  layer_dropout(rate = drop_out_4) %>%
  layer_dense(units = n_output, activation = o_ativate)

# Compile model
model %>%
  compile(loss = loss_func,
          optimizer = optimizer_rmsprop(lr = learn_rate),
          metrics = c('accuracy')
)

# View model
model %>% summary %>% print

# Train model
# ------------------------------------------------------------------------------
history = model %>%
  fit(x = X_train,
      y = y_train,
      epochs = n_epochs,
      batch_size = batch_size,
      validation_split = 0
)

# Evaluate model
# ------------------------------------------------------------------------------
perf_test = model %>% evaluate(X_test, y_test)
acc_test = perf_test %>% pluck('acc') %>% round(3) * 100
perf_train = model %>% evaluate(X_train, y_train)
acc_train = perf_train %>% pluck('acc') %>% round(3) * 100
results = bind_rows(
  tibble(y_true = y_test %>%
           apply(1, function(x){ return( which(x==1) - 1) }) %>%
           factor,
         y_pred = model %>%
           predict_classes(X_test) %>%
           factor,
         Correct = ifelse(y_true == y_pred ,"yes", "no") %>%
           factor,
         data_type = 'test')
  ,
  tibble(y_true = y_train %>%
           apply(1, function(x){ return( which(x==1) - 1) }) %>%
           factor,
         y_pred = model %>%
           predict_classes(X_train) %>%
           factor,
         Correct = ifelse(y_true == y_pred ,"yes", "no") %>%
           factor,
         data_type = 'train'))
my_counts = results %>% count(y_pred, y_true, data_type)

# Visualise model performance
# ------------------------------------------------------------------------------
title = paste0('Performance of Deep Feed Forward Neural Network (',
               'Total number of model parameters = ', count_params(model), ').')
sub_title = paste0("Test Accuracy = ", acc_test, "%, n = ", nrow(X_test), ". ",
                   "Training Accuracy = ", acc_train, "%, n = ", nrow(X_train), ".")
xlab  = 'Predicted (Class assigned by Keras/TensorFlow deep FFN)'
ylab  = 'Measured (Real class, as predicted by netMHCpan-4.0)'
results %>%
  ggplot(aes(x = y_pred, y = y_true, fill = Correct)) +
  geom_jitter(pch = 21, size = 4, alpha = 0.4, colour = 'black') +
  geom_text(data = my_counts, aes(x = y_pred, y = y_true, label = n),
            size = 20, inherit.aes = FALSE) +
  xlab(xlab) +
  ylab(ylab) +
  ggtitle(label = title, subtitle = sub_title) +
  theme_bw() +
  theme(legend.position = "bottom") +
  scale_color_manual(labels = c('No', 'Yes'),
                     values = c('tomato','cornflowerblue')) +
  facet_wrap(~data_type, nrow = 1)

# Save model
# ------------------------------------------------------------------------------
save_model_hdf5(object = model,
                filepath = "Models/05_peptide_model.h5")
