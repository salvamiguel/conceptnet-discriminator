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

def digest(output_lock, output_path, min_cosine, max_jumps, line_data):
    
    position = line_data[0]
    rel_1 = calculate_result(a_star_threads(w1=line_data[1], w2=line_data[3], start_cosine=min_cosine, max_jumps=max_jumps))
    rel_2 = calculate_result(a_star_threads(w1=line_data[2], w2=line_data[3], start_cosine=min_cosine, max_jumps=max_jumps))
    r = [position, line_data[1], line_data[2], line_data[3], rel_1, rel_2, int(line_data[4])]


    output_lock.acquire()
    try:
        w = open(output_path, 'a+')
        w.write(json.dumps(r) + "\n")
        w.flush()
        os.fsync(w)
    finally:
        output_lock.release()

def digest_data_threads(input_path, output_path, min_cosine=0.2, max_jumps = 3, num_proc=None):
    if num_proc == None:
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

    func = partial(digest, output_lock, output_path, min_cosine, max_jumps)

    with tqdm(total=len(lines)) as pbar:
            for i, _ in enumerate(pool.imap(func, lines)):
                pbar.update()



    pool.close()
    pool.join()

argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--input', help='Path to input file. // Ruta del archivo de entrada.')
argparser.add_argument('-o', '--output', help='Path to output file. // Ruta del archivo de salida.')
argparser.add_argument('-c', '--cosine', help='Min cosine in relations. // Minimos coseno en relaciones.', default=0.2)
argparser.add_argument('-j', '--jumps', help='Max jumps in relations. // Número máximo de saltos relaciones.', default=3)


argparser.add_argument('-p', '--processes', help='Number of processes. // Numero de procesos.')

args = argparser.parse_args()

digest_data_threads(input_path=args.input, output_path=args.output, min_cosine=float(args.cosine), max_jumps=int(args.jumps))