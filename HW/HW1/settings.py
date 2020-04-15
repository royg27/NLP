
def dont_use_vectorized():
    global use_vectorized_sparse
    use_vectorized_sparse = False

def use_vectorized():
    global use_vectorized_sparse
    use_vectorized_sparse = True