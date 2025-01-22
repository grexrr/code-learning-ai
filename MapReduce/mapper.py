import sys

class Mapper:
    def __init__(self) -> None:
        self.word_count = {}
    
    def clean_word(self, word):   
        word = word.lower()
        return ''.join(c for c in word if c.isalnum())
    
    def map(self):
        for line in sys.stdin:
            line = line.strip()
            words = line.split()
            
            for word in words:
                cleaned_word = self.clean_word(word)
                if cleaned_word: 
                    print(f'{cleaned_word}\t1')

if __name__ == "__main__":
    mapper = Mapper()
    mapper.map()