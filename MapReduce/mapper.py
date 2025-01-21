import sys

def mapper():
    for line in sys.stdin:
        line = line.strip()
        words = line.split()    
        for word in words:
            print(f'{word}\t1')

if __name__ == "__main__":
    mapper()