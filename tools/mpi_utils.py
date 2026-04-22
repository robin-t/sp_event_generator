try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1

from tqdm import tqdm

# -----------------------------------------
# Basic helpers
# -----------------------------------------

def is_root():
    return RANK == 0


def root_print(*args, **kwargs):
    if RANK == 0:
        print(*args, **kwargs)


def root_input(prompt):
    if RANK == 0:
        value = input(prompt)
    else:
        value = None

    if COMM:
        value = COMM.bcast(value, root=0)

    return value


def bcast(value):
    if COMM:
        return COMM.bcast(value, root=0)
    return value


def gather(value):
    if COMM:
        return COMM.gather(value, root=0)
    return [value]


def barrier():
    if COMM:
        COMM.Barrier()

def mpi_tqdm(iterable, **kwargs):
    if RANK == 0:
        return tqdm(iterable, **kwargs, dynamic_ncols=True)
    return iterable

# -----------------------------------------
# Work distribution helper
# -----------------------------------------

def distribute_indices(n_items):
    """
    Returns the indices assigned to this rank.
    """
    return range(n_items)[RANK::SIZE]