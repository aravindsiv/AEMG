import os 

def check_number_of_attractors(fname):
    found_attractors = -1
    with open(fname) as f:
        line = f.readline()
        # Obtain the last number after a comma
        found_attractors = int(line.split(",")[-1])
    return found_attractors

if __name__ == "__main__":
    test_fname = "/common/home/as2578/repos/AEMG/examples/output/pendulum_lqr1k/0d2c1276d7a742b3854786af892ea3ed.txt"
    print(check_number_of_attractors(test_fname))

    
