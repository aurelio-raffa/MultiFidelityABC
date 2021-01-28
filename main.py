from source.examples.poisson_multifidelity import main
from source.utils import diagnostics, misc

if __name__ == '__main__':
    diagnostics.root_dir = './images/'
    misc.root_dir = './outputs/'
    main()
