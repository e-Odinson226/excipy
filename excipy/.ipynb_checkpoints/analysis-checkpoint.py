import numpy as np
import matplotlib.pyplot as plt
import h5py
from  scipy.stats import linregress,gaussian_kde
import collections



# coords array you already build
#coords = np.array([mol.get_center_of_mass() for mol in mol])
# -------------------------------------------------------------

# ---- helper --------------------------------------------------
def _build_traj_dict(times, records):
    """
    records[step] = [(uid, site, imx, imy, imz), ...]   ### PBC MOD
    returns uid -> (np.array(t), np.array(site_idx), np.array(img, shape=(N,3)))
    """
    traj = {}
    for t, row in zip(times, records):
        for uid, site, ix, iy, iz in row:               ### PBC MOD
            if uid not in traj:
                traj[uid] = ([], [], [])
            traj[uid][0].append(t)
            traj[uid][1].append(site)
            traj[uid][2].append((ix, iy, iz))           ### PBC ADD
    # convert to numpy
    for uid in traj:
        t_list, s_list, img_list = traj[uid]
        traj[uid] = (np.asarray(t_list, dtype=float),
                     np.asarray(s_list, dtype=int),
                     np.asarray(img_list, dtype=int))
    return traj


# ---- single‑run MSD -----------------------------------------
def msd_single_run(times, records, coords, t_grid=None, box=None):  ### PBC ADD box
    """
    box : (Lx, Ly, Lz) or None - if given, unwrap positions
    """
    coords = np.asarray(coords)
    if t_grid is None:
        t_grid = np.asarray(times)

    traj = _build_traj_dict(times, records)
    msd_accum = np.zeros_like(t_grid, dtype=float)
    counts    = np.zeros_like(t_grid, dtype=int)

    for t_vec, site_vec, img_vec in traj.values():       ### PBC MOD
        # reference position (unwrapped)
        if box is None:
            r0 = coords[site_vec[0]]
            r  = coords[site_vec]
        else:                               ### PBC ADD >>>
            shift = img_vec * np.asarray(box)            # shape (N,3)
            r0 = coords[site_vec[0]] + shift[0]
            r  = coords[site_vec] + shift           ### PBC ADD <<<

        disp2 = ((r - r0) ** 2).sum(axis=1)

        idx = np.searchsorted(t_grid, t_vec, side='right') - 1
        idx[idx < 0] = 0
        np.add.at(msd_accum, idx, disp2)
        np.add.at(counts,    idx, 1)

    mask = counts > 0
    msd = np.zeros_like(msd_accum)
    msd[mask] = msd_accum[mask] / counts[mask]
    return t_grid, msd


#  average over runs 
def msd_all_runs(all_runs, coords, n_bins=500, box=None):          ### PBC ADD box
    t_max = max(times[-1] for times, _ in all_runs)
    t_grid = np.linspace(0.0, t_max, n_bins)

    msd_accum = np.zeros(n_bins)
    counts    = np.zeros(n_bins, dtype=int)

    for times, recs in all_runs:
        _, msd_local = msd_single_run(times, recs, coords, t_grid, box=box)  ### PBC MOD
        mask = msd_local > 0
        msd_accum[mask] += msd_local[mask]
        counts[mask]    += 1

    msd_all = np.zeros_like(msd_accum)
    ok = counts > 0
    msd_all[ok] = msd_accum[ok] / counts[ok]
    return t_grid, msd_all




def plot_msd_with_fit(t, msd,
                      t_min=None, t_max=None,
                      title=None,
                      ax=None,
                      scale=100):
    """
    MSD plott with linear fitting 
   
    """

    if ax is None:
        plt.rcParams.update({"font.size": 15})
        fig, ax = plt.subplots(figsize=(6,4.5))

    ax.plot(t, msd/scale, label="MSD")

    # ------- choose fit window ---------------------------------
    if t_min is None:   t_min = float(t.min())
    if t_max is None:   t_max = float(t.max())

    mask = (t >= t_min) & (t <= t_max)
    if mask.sum() < 2:
        raise ValueError("Fit window has < 2 points")

    # ------- linear regression ---------------------------------
    slope, intercept, r, p, stderr = linregress(t[mask], msd[mask])
    D = slope / 6.0            # 3‑D:  MSD = 6 D t

    # fitted line
    t_fit  = np.array([t_min, t_max])
    msd_fit = slope * t_fit + intercept
    ax.plot(t_fit, msd_fit/scale, 'r--', lw=2, label="linear fit")

    # ------- annotation ----------------------------------------
    txt = (f"Slope = {slope*1e-2:.0f} ± {stderr*1e-2:.0f}\n"
           f"   D = {D*1e-5:.1f}  [$ nm^2/ps$]")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", alpha=0.25))

    # ------- cosmetics -----------------------------------------
    ax.set_xlabel("time  [ns]")
    ax.set_ylabel(f"MSD [$nm^2$]")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    return slope, stderr, D



# ────────────────────────────────────────────────────────────────
def dump_output(all_runs, filename="kmc.h5"):
    """
    Stores each trajectory under /traj<i>/time  and  /traj<i>/rec.
    Each step row is encoded as
        "uid,site,imx,imy,imz;uid,site,imx,imy,imz;..."
    """
    with h5py.File(filename, "w") as f:
        f.attrs["n_traj"] = len(all_runs)
        for tidx, (times, recs) in enumerate(all_runs):
            g = f.create_group(f"traj{tidx}")
            g.create_dataset("time", data=np.asarray(times, dtype=np.float64),
                             compression="gzip", compression_opts=4)

            flat_rows = []
            for step in recs:
                row_str = ";".join(f"{u},{s},{ix},{iy},{iz}"
                                   for u, s, ix, iy, iz in step)
                flat_rows.append(row_str.encode())
            g.create_dataset("rec", data=flat_rows,
                             dtype=h5py.string_dtype(), compression="gzip")

    print(f"Wrote {len(all_runs)} trajectories to {filename}")

# ────────────────────────────────────────────────────────────────
def load_output(filename="kmc.h5"):
    """Returns list of (times, records) with 5‑tuple records."""
    runs = []
    with h5py.File(filename, "r") as f:
        n_traj = f.attrs["n_traj"]
        for tidx in range(n_traj):
            g  = f[f"traj{tidx}"]
            t  = g["time"][:]

            rec = []
            for row in g["rec"][:]:
                if len(row) == 0:
                    rec.append([])
                else:
                    tuples = []
                    for pair in row.decode().split(";"):
                        parts = list(map(int, pair.split(",")))
                        if len(parts) == 2:
                            uid, site = parts
                            tuples.append((uid, site, 0, 0, 0))  # legacy file
                        else:
                            uid, site, ix, iy, iz = parts
                            tuples.append((uid, site, ix, iy, iz))
                    rec.append(tuples)
            runs.append((t, rec))
    return runs




def load_output_light(filename="kmc.h5"):
    """
    Loads both legacy records (uid, site) and new PBC records
    (uid, site, ix, iy, iz).  Always returns 5‑column int32 arrays.
    """
    runs = []

    with h5py.File(filename, "r") as f:
        n_traj = f.attrs["n_traj"]

        for tidx in range(n_traj):
            g     = f[f"traj{tidx}"]
            times = g["time"][:]            # float64

            step_arrays = []
            for row in g["rec"]:
                if len(row) == 0:
                    step_arrays.append(np.empty((0, 5), dtype=np.int32))
                    continue

                # row is already bytes → decode once
                tokens = row.decode().split(";")
                n_cols = len(tokens[0].split(","))       # 2 or 5

                ints = np.fromiter(
                          (int(x) for tok in tokens for x in tok.split(",")),
                          dtype=np.int32, count=len(tokens)*n_cols
                       ).reshape(-1, n_cols)

                if n_cols == 2:                           # pad legacy file
                    pad  = np.zeros((ints.shape[0], 3), dtype=np.int32)
                    ints = np.hstack((ints, pad))

                step_arrays.append(ints)

            runs.append((times, step_arrays))

    return runs




def plot_average_exciton_population(results, num_bins=100,x0=0, xf=0.8):
    """
    Plot the average number of excitons vs. time over multiple KMC trajectories.

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
    plt.figure(figsize=(5,4))
    plt.rcParams.update({"font.size": 14})
    
    plt.plot(t_grid, avg_excitons_per_bin, '-', color="b", linewidth=2)
    plt.scatter(t_grid, avg_excitons_per_bin, marker="o", color="green", linewidth=0.1)
    plt.xlim(x0,xf)
    plt.xlabel("Time [ns]")
    plt.ylabel("Exciton density")
    #plt.title(f"Average Exciton Population vs. Time ({n_trajs} Trajectories)")
    return t_grid, avg_excitons_per_bin

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
       # plt.show()




# ────────────────────────────────────────────────────────────────
def diffusion_length(all_runs,
                      coords,
                      box=None,
                      bins=40,
                      kind='hist',# 'hist' or 'kde'
                      include_alive=False,
                      plot=True,
                      ax=None):
    """
    Computes displacement for every *individual* exciton
    in every trajectory.  

    Returns:
    lengths 
    stats{mean, median, rms, n}
    """
    coords = np.asarray(coords)
    box    = np.asarray(box) if box is not None else None

    r_init  = {}        # (traj,uid) -> r0
    r_final = {}        # (traj,uid) -> rN
    alive   = set()     # keys still present at last step

    for traj_id, (times, recs) in enumerate(all_runs):
        prev = {uid:(site,ix,iy,iz) for uid,site,ix,iy,iz in recs[0]}
        for row in recs:
            curr = {uid:(site,ix,iy,iz) for uid,site,ix,iy,iz in row}
            # first appearance
            for uid,(site,ix,iy,iz) in curr.items():
                key = (traj_id, uid)
                if key not in r_init:
                    r0  = coords[site]
                    if box is not None:
                        r0 += np.array([ix,iy,iz])*box
                    r_init[key] = r0
            # update last position every step
            for uid,(site,ix,iy,iz) in curr.items():
                key = (traj_id, uid)
                r   = coords[site]
                if box is not None:
                    r += np.array([ix,iy,iz])*box
                r_final[key] = r
            prev = curr
        alive.update((traj_id,uid) for uid in prev)   # the ones still present

    if not include_alive:
        for key in alive:
            r_init.pop(key, None)
            r_final.pop(key, None)

    vecs = [r_final[k] - r_init[k] for k in r_final if k in r_init]
    if not vecs:
        raise ValueError("No completed exciton paths found: (maybe all are still alive).")

    lengths = np.linalg.norm(np.vstack(vecs), axis=1)/10

    stats = dict(mean=np.mean(lengths),
                 median=np.median(lengths),
                 rms=np.sqrt(np.mean(lengths**2)),
                 n=len(lengths))

    #plot 
    if plot:
        if ax is None:
            plt.rcParams.update({"font.size": 14})
            fig, ax = plt.subplots(figsize=(5,4))

        if kind == 'hist':
            ax.hist(lengths, bins=bins, density=True,
                    alpha=0.7, edgecolor='k', label='hist')
        else:  # KDE
            xs  = np.linspace(0, lengths.max()*1.05, 400)
            kde = gaussian_kde(lengths)
            ax.plot(xs, kde(xs), lw=2, label='KDE')

        ax.axvline(stats['mean'], color='r', ls='--', lw=1.5,
                   label=f"mean = {stats['mean']:.2f}")
        ax.axvline(stats['rms'],  color='g', ls='--', lw=1.5,
                   label=f"rms  = {stats['rms']:.2f}")

        ax.set_xlabel("diffusion length [$nm$]")
        ax.set_ylabel("probability density")
        ax.legend(); ax.grid(alpha=0.4)

    return lengths, stats


#CODE DEBUGGING HELPERS

def largest_hops(times, records, coords, box, cutoff=20):
    coords = np.asarray(coords); box = np.asarray(box)
    hop_map = {}     # uid -> list of hop magnitudes

    prev_row = {uid:(site,ix,iy,iz) for uid,site,ix,iy,iz in records[0]}
    for row in records[1:]:
        for uid,site,ix,iy,iz in row:
            if uid in prev_row:
                psite,pix,piy,piz = prev_row[uid]
                r0 = coords[psite] + np.array([pix,piy,piz])*box
                r1 = coords[site]  + np.array([ix,iy,iz])*box
                d  = np.linalg.norm(r1-r0)
                hop_map.setdefault(uid, []).append(d)
            prev_row[uid] = (site,ix,iy,iz)

    big = [(uid,max(ds)) for uid,ds in hop_map.items() if max(ds) > cutoff+1e-3]
    print(f"Total uids checked: {len(hop_map)}")
    if big:
        print("Hops larger than cutoff:")
        for uid,d in sorted(big, key=lambda x: -x[1])[:10]:
            print(f"  uid {uid:4}:  max hop = {d:6.2f} Å")
    else:
        print("No hop exceeds cutoff.")



def find_large_hops(times, records, coords, box, cutoff=20.0, max_print=20):
    """
   
    """
    coords = np.asarray(coords)
    box    = np.asarray(box)
    bad_steps = []

    # uid -> (site, ix, iy, iz, r_vec)
    last_pos = {uid:(site,ix,iy,iz,
                     coords[site] + np.array([ix,iy,iz])*box)
                for uid,site,ix,iy,iz in records[0]}

    for step, (t, row) in enumerate(zip(times[1:], records[1:]), start=1):
        for uid,site,ix,iy,iz in row:
            if uid not in last_pos:    # new exciton born
                last_pos[uid] = (site,ix,iy,iz,
                                 coords[site] + np.array([ix,iy,iz])*box)
                continue

            p_site,p_ix,p_iy,p_iz, r_prev = last_pos[uid]
            r_now = coords[site] + np.array([ix,iy,iz])*box
            d = np.linalg.norm(r_now - r_prev)

            if d > cutoff:
                bad_steps.append(dict(step=step,
                                      time_ns=t,
                                      uid=uid,
                                      from_pos=(p_site,p_ix,p_iy,p_iz),
                                      to_pos  =(site, ix,iy,iz),
                                      shift   =(ix-p_ix, iy-p_iy, iz-p_iz),
                                      hop_len=d))
                if len(bad_steps) <= max_print:
                    info = bad_steps[-1]
                    print(f"step {info['step']:6}  t = {info['time_ns']:8.3f} ns  "
                          f"uid {uid:4}  hop = {info['hop_len']:7.2f} Å  "
                          f"shift {info['shift']}")
            # update
            last_pos[uid] = (site,ix,iy,iz,r_now)

    print(f"\nTotal hops > {cutoff} Å : {len(bad_steps)}")
    return bad_steps
