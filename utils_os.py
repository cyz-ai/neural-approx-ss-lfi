import os.path
import os
import pickle
import numpy as np
import inspect
import multiprocessing, time



def get_filename(algorithm):
    '''
    Creates a unique filename for this algorithm.
    Arguments
    ---------
    algorithm : instance of an ABC algorithm
        The algorithm instance to generate a name for.
    '''

    return str(algorithm) + '.abc'


def load_algorithm(DIR, algorithm):
    '''
    Loads the result_data of the given algorithm with given parameters for
    the given problem. If there are no result data for these combinations, None is
    returned.
    Parameters
    ----------
    algorithm :
        An instance of an ABC algorithm
    Returns
    -------
    result : instance of an ABC algorithm
    '''

    filename = get_filename(algorithm)

    dir_path = os.path.join(os.getcwd(), '{}/algorithm'.format(DIR))
    file_path = os.path.join(dir_path, filename)
     
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            result = pickle.load(f)
        return result
    else:
        return None

def save_algorithm(DIR, algorithm):
    '''
    Saves the result_data of the given algorithm with given parameters for
    the given problem.
    Parameters
    ----------
    algorithm :
        An instance of an ABC algorithm
    '''

    filename = get_filename(algorithm)
    dir_path = os.path.join(os.getcwd(), '{}/algorithm'.format(DIR))
    file_path = os.path.join(dir_path, filename)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_path, 'wb') as f:
        pickle.dump(algorithm, f)

def save_object(DIR, filename, obj):
    dir_path = os.path.join(os.getcwd(), '{}/data'.format(DIR))
    file_path = os.path.join(dir_path, filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.save(file_path, obj)

def load_object(DIR, filename):
    dir_path = os.path.join(os.getcwd(), '{}/data'.format(DIR))
    file_path = os.path.join(dir_path, filename)
    if not os.path.exists(dir_path):
        return None
    else:
        return np.load(file_path)

def is_file_exist(DIR, filename):
    dir_path = os.path.join(os.getcwd(), '{}/data'.format(DIR))
    full_path = os.path.join(dir_path, filename)
    return os.path.exists(full_path)

def run_in_parallel(func, params_per_proc, N_proc):
    rets = None
    print('# of cpus = ', N_proc)
    with multiprocessing.Pool(N_proc) as p:
        rets = p.map(func, params_per_proc)
        p.close()
    return rets

def set_cpu_affinity(cpu_no):
    pid = os.getpid()
    os.system('taskset -p {0} {1}'.format(cpu_no, pid))
    return

def get_n_cpus():
    return multiprocessing.cpu_count()