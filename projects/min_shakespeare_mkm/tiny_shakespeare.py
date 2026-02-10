# tiny shakespeare 
import random, re
from collections import defaultdict

txt = open("shakespeare.txt", encoding="utf-8").read().lower()
words = re.findall(r"[a-z'’]+|[.,;:!?]", txt)

chain = defaultdict(list)
for i in range(len(words) - 2):
    chain[(words[i], words[i+1])].append(words[i+2])

w1, w2 = random.choice(list(chain))
out = [w1, w2]
for _ in range(200):
    nxt = random.choice(chain[(w1, w2)])
    out.append(nxt)
    w1, w2 = w2, nxt

print(" ".join(out).replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?"))

