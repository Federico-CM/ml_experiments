import csv, math, re

# Tokenizer: extract lowercase words (letters, numbers, apostrophes)
tok = lambda s: re.findall(r"[a-z0-9']+", s.lower())

# Counters for spam/ham message counts
spam = ham = 0

# Word frequency dictionaries for spam and ham
ws = {}   # word counts in spam
wh = {}   # word counts in ham

# Read dataset (label, message)
for y, m in csv.reader(open(
    "SMSSpamCollection.csv",
    encoding="utf-8"
)):
    
    y = 1 if y.strip() == "spam" else 0
    
    # Choose correct dictionary
    target = ws if y else wh
    
    # Manually update counts
    for w in tok(m):
        target[w] = target.get(w, 0) + 1
    
    spam += y
    ham += 1 - y


# Vocabulary size
V = len(set(ws.keys()) | set(wh.keys()))

# Laplace smoothing parameter
a = 1

# Log priors
ps = math.log(spam / (spam + ham))
ph = math.log(ham / (spam + ham))

# Log denominators (with smoothing)
ls = math.log(sum(ws.values()) + a * V)
lh = math.log(sum(wh.values()) + a * V)


def is_spam(msg):
    
    s = ps
    h = ph
    
    for w in tok(msg):
        s += math.log(ws.get(w, 0) + a) - ls
        h += math.log(wh.get(w, 0) + a) - lh
    
    return s > h, s - h


# Interactive loop
while True:
    t = input("> ")
    p, margin = is_spam(t)
    print(("spam" if p else "not spam"), "margin:", round(margin, 3))

 
