if __name__ == "__main__":
    import time
    from prospect.fitting import fit_model
    from prospect.io import write_results as writer
    from prospect import prospect_args

    # Get the default argument parser
    parser = prospect_args.get_parser()
    # Add custom arguments that controll the build methods
    parser.add_argument("--custom_argument_1", ...)
    # Parse the supplied arguments, convert to a dictionary, and add this file for logging purposes
    args = parser.parse_args()
    run_params = vars(args)
    run_params["param_file"] = __file__

    # Build the fit ingredients on each process
    obs, model, sps, noise = build_all(**run_params)
    run_params["sps_libraries"] = sps.ssp.libraries

    # Set up MPI communication
    try:
        import mpi4py
        from mpi4py import MPI
        from schwimmbad import MPIPool

        mpi4py.rc.threads = False
        mpi4py.rc.recv_mprobe = False

        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        withmpi = comm.Get_size() > 1
    except ImportError:
        print('Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.')
        withmpi = False

# Evaluate SPS over logzsol grid in order to get necessary data in cache/memory
# for each MPI process. Otherwise, you risk creating a lag between the MPI tasks
# caching data depending which can slow down the parallelization
if (withmpi) & ('logzsol' in model.free_params):
    dummy_obs = dict(filters=None, wavelength=None)

    logzsol_prior = model.config_dict["logzsol"]['prior']
    lo, hi = logzsol_prior.range
    logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)

    sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
    for logzsol in logzsol_grid:
        model.params["logzsol"] = np.array([logzsol])
        _ = model.predict(model.theta, obs=dummy_obs, sps=sps)

# ensure that each processor runs its own version of FSPS
# this ensures no cross-over memory usage
from prospect.fitting import lnprobfn
from functools import partial
lnprobfn_fixed = partial(lnprobfn, sps=sps)

if withmpi:
    run_params["using_mpi"] = True
    with MPIPool() as pool:

        # The dependent processes will run up to this point in the code
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        nprocs = pool.size
        # The parent process will oversee the fitting
        output = fit_model(obs, model, sps, noise, pool=pool, queue_size=nprocs, lnprobfn=lnprobfn_fixed, **run_params)
else:
    # without MPI we don't pass the pool
    output = fit_model(obs, model, sps, noise, lnprobfn=lnprobfn_fixed, **run_params)

# Set up an output file and write
ts = time.strftime("%y%b%d-%H.%M", time.localtime())
hfile = "{0}_{1}_mcmc.h5".format(args.outfile, ts)
writer.write_hdf5(hfile, run_params, model, obs,
                  output["sampling"][0], output["optimization"][0],
                  tsample=output["sampling"][1],
                  toptimize=output["optimization"][1],
                  sps=sps)

try:
    hfile.close()
except(AttributeError):
    pass