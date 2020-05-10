from conceptnet import *

def read_from_file(file):
    """Reads data from file to digest and prepare it to the neural network.
    File format has to match with the following: <word1>, <word2>, <attribute>, <0 | 1 whether if attribute is discriminative>

    Arguments:
        file {str} -- File path

    Returns:
        list -- List of tuples containing the data from file
    """
    r = open(file, "r")
    lines = r.readlines()
    data = []

    for line in lines:
        data.append(tuple() + tuple(i for i in line.replace("\n", "").split(',')))
    return data

def digest(output_lock, output_path, line_data):
    
    position = line_data[0]
    rel_1 = calculate_result(a_star_threads(line_data[1], line_data[3]))
    rel_2 = calculate_result(a_star_threads(line_data[2], line_data[3]))
    r = [position, line_data[1], line_data[2], line_data[3], rel_1, rel_2, int(line_data[4])]


    output_lock.acquire()
    try:
        w = open(output_path, 'a+')
        w.write(json.dumps(r) + "\n")
        w.flush()
        os.fsync(w)
    finally:
        output_lock.release()

def digest_data_threads(input_path, output_path, num_proc=False):
    if not num_proc:
        num_proc = os.cpu_count()
    tuples = read_from_file(input_path)
    pos = 0
    pool = Pool(processes=num_proc)
    m = Manager()
    output_lock = m.Lock()
    lines = []
    for line in tuples:
        l_pos = (pos,)
        l = l_pos + line
        lines.append(l)
        pos = pos + 1

    func = partial(digest, output_lock, output_path)

    with tqdm(total=len(lines)) as pbar:
            for i, _ in enumerate(pool.imap(func, lines)):
                pbar.update()



    pool.close()
    pool.join()

argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--input', help='Path to input file. // Ruta del archivo de entrada.')
argparser.add_argument('-o', '--output', help='Path to output file. // Ruta del archivo de salida.')
argparser.add_argument('-p', '--processes', help='Number of processes. // Número de procesos.')

args = argparser.parse_args()

digest_data_threads(args.input, args.output)