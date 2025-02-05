from collections import defaultdict

add = lambda x, y: x + y

print(add(4, 6))

numbers = [1, 3, 4, 2, 6]
new_numbers = list(map(lambda x : 2 * x, numbers))
print(new_numbers)