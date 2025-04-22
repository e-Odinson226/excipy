import numpy as np
import matplotlib.pyplot as plt


def plot_average_exciton_population(results, num_bins=100,x0=0, xf=0.6):
    """
    Plot the average number of excitons vs. time across multiple KMC trajectories.

    Parameters
    ----------
    results : list of (times, exciton_records)
        Each element is one trajectory's data:
          - times: list of float (the event times)
          - exciton_records: list of lists, where exciton_records[i]
            is the list of site indices for all excitons at times[i].
        Usually times[i] and exciton_records[i] have the same length.
    num_bins : int
        Number of points (bins) in the common time grid for averaging.
    """

    # 1) Find the maximum end time among all trajectories
    all_end_times = [traj_times[-1] for (traj_times, ex_records) in results if len(traj_times) > 0]
    if not all_end_times:
        print("No data in results.")
        return

    t_max = max(all_end_times)

    # 2) Create a common time grid from 0 to t_max
    t_grid = np.linspace(0, t_max, num_bins)

    # This array will accumulate the total exciton count across all trajectories, for each bin
    total_excitons_per_bin = np.zeros(num_bins, dtype=float)

    # 3) For each trajectory, step through the time grid
    n_trajs = len(results)

    for (traj_times, ex_records) in results:
        if len(traj_times) == 0:
            # no events here, skip
            continue

        # For each time bin t_grid[i], find how many excitons are present
        # We'll do a step-function approach: pick the last known event time <= t_grid[i].
        # np.searchsorted can help:
        #   idx = searchsorted(traj_times, t, side='right') - 1
        # If idx < 0, we haven't started => so we can treat it as ex_records[0] or 0 excitons
        # as you prefer (by default we assume the first record applies at t=0).
        for i, t_val in enumerate(t_grid):
            idx = np.searchsorted(traj_times, t_val, side='right') - 1
            if idx < 0:
                # before the first recorded time => assume excitons = ex_records[0]
                idx = 0

            # The number of excitons at that step is just len(ex_records[idx])
            n_exc = len(ex_records[idx])
            total_excitons_per_bin[i] += n_exc

    # 4) Average across all trajectories
    avg_excitons_per_bin = total_excitons_per_bin / n_trajs

    # 5) Plot
    plt.figure()
    plt.plot(t_grid, avg_excitons_per_bin, '-', color="b", linewidth=2)
    #plt.scatter(t_grid, avg_excitons_per_bin, marker="o", color="green", linewidth=0.1)
    plt.xlim(x0,xf)
    plt.xlabel("Time (ns)")
    plt.ylabel("Exciton density")
    plt.title(f"Average Exciton Population vs. Time ({n_trajs} Trajectories)")
    plt.show()


    def plot_exciton_density(times, exciton_records, num_sites):
        """
        Plot exciton occupancy vs. time.
    
        Parameters
        ----------
        times : list of float
            The KMC times after each event (length T).
        exciton_records : list of list of int
            exciton_records[t] is a list of site indices occupied by excitons at time[t].
        num_sites : int
            Number of lattice sites in the simulation.
        """
    
        # 1) Build a 2D array [time_index, site_index] = number of excitons at site
        T = len(times)
        density = np.zeros((T, num_sites), dtype=int)
    
        for t_idx, ex_sites in enumerate(exciton_records):
            for site_idx in ex_sites:
                density[t_idx, site_idx] += 1
    
        # 2) Plot total exciton population vs. time
        total_excitons = density.sum(axis=1)  # sum over all sites at each time
    
        plt.figure()
        plt.plot(times, total_excitons, label='Total excitons')
        plt.xlabel('Time (ns)')
        plt.ylabel('Number of excitons')
        plt.title('Total Exciton Population vs. Time')
        plt.legend()
        plt.show()