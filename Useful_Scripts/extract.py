

def extract_actual_alternate(filename='../td_error_debug.txt', cycles=20):
	f = open(filename, 'r')
	lines = f.readlines()
	numbers = []
	for line in lines:
		numbers.append(float(line.strip().split()[-1]))
	return numbers[::cycles]