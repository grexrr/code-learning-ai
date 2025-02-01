from collections import defaultdict

a = [("k", (3, 4)), ("m", (5, 3))]

array = defaultdict(list)
for k, v in a:
    array[k]=(v)

print(array)



