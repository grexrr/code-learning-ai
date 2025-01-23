import sys

class WordCountMapper:
    def __init__(self) -> None:
        pass
    
    def process(self, word):
        print("processing: " + word)
        return word
    
    def mapping(self):
        for line in sys.stdin:
            words = line.strip().split()
            for word in words:
                w = self.process(word)
                pair = (w, 1)
                print(f"Emitting: {pair}")
                yield pair
            print("-------------------------------------")

# ... rest of the code remains the same ...

    # def mapping(self, str):
    #     for line in str:
    #         words = line.strip().split()
    #         for word in words:
    #             w = self.process(word)
    #             yield((w, 1))
    #         print("-------------------------------------")

    def run(self):
        return self.mapping()
    
    # def test(self):
    #     test_input = """Hello World
    #     Hello MapReduce
    #     This is a test
    #     World is beautiful"""
    #     print(list(self.mapping(test_input)))


if __name__ == "__main__":
    mapper = WordCountMapper()
    for key_value_pair in mapper.run():
        print(f"Output: {key_value_pair}")
