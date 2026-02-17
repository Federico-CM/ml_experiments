# Tiny (and fast) Shakespeare text gen using a markov chain
library(stringi)
library(data.table)
#set.seed(0)
txt <- stri_trans_tolower(paste(readLines("shakespeare.txt", warn=FALSE), collapse=" "))
words <- unlist(stri_extract_all_regex(txt, "[a-z'’]+|[.,;:!?]"))

DT <- data.table(
  k1 = words[1:(length(words)-3)],
  k2 = words[2:(length(words)-2)],
  k3 = words[3:(length(words)-1)],
  nxt = words[4:length(words)]
)

setkey(DT, k1, k2, k3)

start <- DT[sample(.N,1), .(k1,k2,k3)]
w <- as.character(start)
out <- w

for(i in 1:200){
  row <- DT[.(w[1], w[2], w[3])]
  nxt <- sample(row$nxt, 1)
  out <- c(out, nxt)
  w <- c(w[-1], nxt)
}

cat(stri_replace_all_regex(paste(out, collapse=" "), " ([,.;:!?])", "$1"), "\n")

