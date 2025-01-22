import sys

class Reducer:
    def __init__(self) -> None:
        self.word_count = {}
    
    def reduce(self, key, values):  
        self.word_count[key] = sum(values)
        return (key, self.word_count[key])
    
    def run(self):
        current_word = None
        current_counts = []

  
        for line in sys.stdin:
            line = line.strip()
            word, count = line.split('\t')
            count = int(count)
  
            if current_word and current_word != word:
                result = self.reduce(current_word, current_counts)
                print(f'{result[0]}\t{result[1]}')
                current_counts = []
            
            current_word = word
            current_counts.append(count)
        
        if current_word:
            result = self.reduce(current_word, current_counts)
            print(f'{result[0]}\t{result[1]}')

if __name__ == "__main__":
    reducer = Reducer()
    reducer.run()