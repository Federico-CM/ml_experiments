# tiny shakespeare (just 20 lines!)
import random, re
from collections import defaultdict

txt = open("shakespeare.txt", encoding="utf-8").read().lower()
words = re.findall(r"[a-z'’]+|[.,;:!?]", txt)

chain = defaultdict(list)
for i in range(len(words) - 3):
    chain[(words[i], words[i+1], words[i+2])].append(words[i+3])

w1, w2, w3 = random.choice(list(chain))
out = [w1, w2, w3]

for _ in range(200):
    nxt = random.choice(chain[(w1, w2, w3)])
    out.append(nxt)
    w1, w2, w3 = w2, w3, nxt

print(" ".join(out).replace(" ,", ",").replace(" .", ".\n").replace(" !", "!").replace(" ?", "?"))


