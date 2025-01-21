def test_environment():
    print("Testing mapper with sample input...")
    
    # test input
    sample_text = """Hello World
    Hello MapReduce
    This is a test"""


    with open('sample_input.txt', 'w') as f:
        f.write(sample_text)
    
    print("Sample input file created. You can now test the mapper using:")
    print("cat sample_input.txt | python mapper.py")

if __name__ == "__main__":
    test_environment()
