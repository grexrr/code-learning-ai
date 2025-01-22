import subprocess
import os

def test_mapreduce():

    test_input = """Hello World
    Hello MapReduce
    This is a test
    World is beautiful"""
    
    # Create output file
    with open('test_input.txt', 'w') as f:
        f.write(test_input)
    
    print("1. Test Mapper Output:")
    
    mapper_cmd = "cat test_input.txt | python mapper.py"
    mapper_output = subprocess.check_output(mapper_cmd, shell=True).decode()
    print(mapper_output)
    
    print("---------------------------------------------------------------")
    print("\n2. Test whole process:")
    full_cmd = "cat test_input.txt | python mapper.py | sort | python reducer.py"
    try:
        full_output = subprocess.check_output(full_cmd, shell=True).decode()
        print("Result:")
        print(full_output)
    except subprocess.CalledProcessError as e:
        print("Error executing MapReduce:", e)
    
    # Clear Test File
    os.remove('test_input.txt')

if __name__ == "__main__":
    test_mapreduce()