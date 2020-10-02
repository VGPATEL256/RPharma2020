# Simple encoding function for converting a given peptide 'x' to a numerical
# representation using the 'm' matrix
encode_peptide = function(x, m){
  X_enc = x %>%
    str_split('') %>%
    lapply(function(x_i){
      m[x_i,] %>%
        as.matrix %>%
        t %>%
        matrix(nrow = 1, byrow = TRUE) %>%
        return
    })
  X_enc = do.call(rbind, X_enc)
  rownames(X_enc) = x
  return(X_enc)
}
