from tabulate import tabulate


def read_file(file_handler):
    fqns = list()

    while True:
        line = file_handler.readline()
        if not line:
            break
        else:
            line = line.strip().split()[-1]
            fqns.append(line)
    return fqns


def get_stats(file_path: str):
	f = open(file_path)
	lines = read_file(f)

	libraries = [
		"com.google.gson",
		"org.apache.commons.logging",
		"org.apache.commons.collections",
		"org.apache.commons.lang3",
		"org.hibernate",
		"com.google.common",
		"org.junit",
		"com.google.gwt",
		"org.joda.time",
		"com.thoughtworks.xstream",
		"android",
		"java"
	]

	unique_fqns = list()
	for library in libraries:
		counter = 0

		lines_with_library = list(filter(lambda line: line.startswith(library), lines))
		unique_fqns.append(len(set(lines_with_library)))
	
	table = list()

	for (library, len_unique) in zip(libraries, unique_fqns):
		row = [library, len_unique]
		table.append(row)
	
	print()
	print("Unique FQNs per library")
	print(tabulate(table))
