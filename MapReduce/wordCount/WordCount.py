def generate_testfile():
    test_input = """Hello World
        Hello MapReduce
        This is a test
        World is beautiful"""
    with open("test_input.txt", 'w') as f:
        f.write(test_input)

def mapper(input_txt):
    for line in input_txt:
        # line = line.strip()
        # print("line: " + line)

        words = line.strip().split()
        print(words)
        for word in words:
            print((word, 1))
            yield (word, 1)
        print("-----------------------------------")

def reducer(word_pairs):
    result = {}
    for word, count in word_pairs:
        if word not in result:
            result[word] = [count]
        else:
            result[word].append(count)
    for word, counts in result.items():
        yield(word, sum(counts))
        
# generate_testfile()
with open("test_input.txt", 'r') as input_file:
    mapped_data = mapper(input_file)
    reduced_data = reducer(mapped_data)

    # print final data
    for word, total in list(reduced_data):
        print(f"{word}: {total}")