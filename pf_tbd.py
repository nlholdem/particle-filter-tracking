"""
Minimal Salmond-Birch particle-filter track-before-detect (PF-TBD).

Reference:
  D. J. Salmond and H. Birch, "A particle filter for track-before-detect",
  Proc. American Control Conference, 2001, pp. 3755-3760.

Model
-----
State per particle:  [x, vx, y, vy, I]  (pixels, pixels/frame, intensity)
Existence flag:      E in {0, 1}, a two-state Markov chain with
                     birth probability P_BIRTH and death probability P_DEATH.
Measurement:         z(i, j) = h(i, j; x) + w,  w ~ N(0, NOISE_STD^2), white
                     h = I * exp(-((j - x)^2 + (i - y)^2) / (2 * PSF_SIGMA^2))
Weight:              likelihood RATIO target-vs-noise, evaluated only over a
                     small patch around each particle (the PSF is local).
                     Particles with E = 0 get ratio 1.
Output:              P(target present) = fraction of particles with E = 1;
                     state estimate = mean of the E = 1 particles.

The input image must already be background-suppressed so the residual noise
is roughly white. For real EO/IR imagery, run motion compensation and
background whitening first (see the notes at the bottom of this file).

Run:  python pf_tbd.py            (writes pf_tbd_results.png)
      python pf_tbd.py --snr 4 --seed 3
"""

import argparse
import time

import numpy as np

# ----------------------------------------------------------------------------
# Scenario and filter parameters
# ----------------------------------------------------------------------------
IMG_SIZE = 64          # image is IMG_SIZE x IMG_SIZE pixels
N_FRAMES = 60
T_APPEAR = 10          # target present for frames [T_APPEAR, T_VANISH)
T_VANISH = 45
PSF_SIGMA = 0.7        # PSF std dev in pixels (target spans about 1-2 px)
NOISE_STD = 1.0

N_PARTICLES = 100000
P_BIRTH = 0.05         # P(E: 0 -> 1) per frame
P_DEATH = 0.02         # P(E: 1 -> 0) per frame
PROC_NOISE_ACC = 0.08  # white-acceleration noise, px/frame^2
PROC_NOISE_INT = 0.05  # intensity random walk, per frame
V_MAX = 1.0            # birth prior on |vx|, |vy|, px/frame
I_BIRTH = (1.0, 5.0)   # birth prior on intensity (uniform)
PATCH_RADIUS = 2       # likelihood evaluated over a (2R+1)^2 patch
DETECT_THRESHOLD = 0.5  # declare "target present" when P(E=1) > this


# ----------------------------------------------------------------------------
# Synthetic data: a dim point target moving in white Gaussian noise
# ----------------------------------------------------------------------------
def make_sequence(snr_db, rng):
    """Return frames (N_FRAMES, H, W), true states (N_FRAMES, 5) and presence mask.

    snr_db is the peak SNR: 20*log10(I / NOISE_STD) for a target centred on
    a pixel. Off-centre targets have a lower peak, which makes it harder.
    """
    intensity = NOISE_STD * 10 ** (snr_db / 20)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    truth = np.full((N_FRAMES, 5), np.nan)
    present = np.zeros(N_FRAMES, dtype=bool)
    frames = rng.normal(0.0, NOISE_STD, size=(N_FRAMES, IMG_SIZE, IMG_SIZE))

    state = np.array([15.3, 0.6, 20.7, 0.45, intensity])  # x, vx, y, vy, I
    for k in range(N_FRAMES):
        if T_APPEAR <= k < T_VANISH:
            present[k] = True
            truth[k] = state
            x, _, y, _, amp = state
            frames[k] += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2)
                                      / (2 * PSF_SIGMA ** 2))
            # gentle manoeuvre halfway through
            if k == (T_APPEAR + T_VANISH) // 2:
                state[1], state[3] = 0.3, 0.75
            state = state + np.array([state[1], 0, state[3], 0, 0])
    return frames, truth, present


# ----------------------------------------------------------------------------
# The filter
# ----------------------------------------------------------------------------
class SalmondBirchPF:
    def __init__(self, n_particles, rng):
        self.n = n_particles
        self.rng = rng
        self.x = np.zeros((n_particles, 5))      # [x, vx, y, vy, I]
        self.e = np.zeros(n_particles, dtype=bool)  # start: no target
        r = np.arange(-PATCH_RADIUS, PATCH_RADIUS + 1)
        self.dy, self.dx = [a.ravel() for a in np.meshgrid(r, r, indexing="ij")]

    # -- birth density: uniform over image, velocity, intensity --------------
    def _sample_birth(self, m):
        out = np.empty((m, 5))
        out[:, 0] = self.rng.uniform(0, IMG_SIZE - 1, m)
        out[:, 2] = self.rng.uniform(0, IMG_SIZE - 1, m)
        out[:, 1] = self.rng.uniform(-V_MAX, V_MAX, m)
        out[:, 3] = self.rng.uniform(-V_MAX, V_MAX, m)
        out[:, 4] = self.rng.uniform(*I_BIRTH, m)
        return out

    # -- prediction: existence Markov chain + constant-velocity motion -------
    def predict(self):
        u = self.rng.random(self.n)
        was = self.e
        born = ~was & (u < P_BIRTH)
        survive = was & (u >= P_DEATH)

        # continuing targets: CV model with white-acceleration noise
        idx = np.flatnonzero(survive)
        m = idx.size
        acc = self.rng.normal(0, PROC_NOISE_ACC, (m, 2))
        xs = self.x[idx]
        xs[:, 0] += xs[:, 1] + 0.5 * acc[:, 0]
        xs[:, 1] += acc[:, 0]
        xs[:, 2] += xs[:, 3] + 0.5 * acc[:, 1]
        xs[:, 3] += acc[:, 1]
        xs[:, 4] = np.abs(xs[:, 4] + self.rng.normal(0, PROC_NOISE_INT, m))
        self.x[idx] = xs

        # newborn targets: draw from the birth density
        b = np.flatnonzero(born)
        self.x[b] = self._sample_birth(b.size)

        self.e = born | survive
        # a target that leaves the image no longer exists in it
        out = (self.x[:, 0] < 0) | (self.x[:, 0] > IMG_SIZE - 1) | \
              (self.x[:, 2] < 0) | (self.x[:, 2] > IMG_SIZE - 1)
        self.e &= ~out

    # -- update: local likelihood ratio, then systematic resampling ----------
    def update(self, z):
        logw = np.zeros(self.n)           # log ratio = 0 for E = 0
        idx = np.flatnonzero(self.e)
        if idx.size:
            px, py, amp = self.x[idx, 0], self.x[idx, 2], self.x[idx, 4]
            cx = np.rint(px).astype(int)[:, None] + self.dx[None, :]
            cy = np.rint(py).astype(int)[:, None] + self.dy[None, :]
            valid = (cx >= 0) & (cx < IMG_SIZE) & (cy >= 0) & (cy < IMG_SIZE)
            cxc = np.clip(cx, 0, IMG_SIZE - 1)
            cyc = np.clip(cy, 0, IMG_SIZE - 1)
            h = amp[:, None] * np.exp(-((cxc - px[:, None]) ** 2 +
                                        (cyc - py[:, None]) ** 2)
                                      / (2 * PSF_SIGMA ** 2))
            zz = z[cyc, cxc]
            # log[ N(z; h, s^2) / N(z; 0, s^2) ] = (2 h z - h^2) / (2 s^2)
            ll = (h * (2 * zz - h)) / (2 * NOISE_STD ** 2)
            logw[idx] = np.sum(np.where(valid, ll, 0.0), axis=1)

        w = np.exp(logw - logw.max())
        w /= w.sum()
        keep = self._systematic_resample(w)
        self.x = self.x[keep]
        self.e = self.e[keep]

    def _systematic_resample(self, w):
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        c = np.cumsum(w)
        c[-1] = 1.0
        return np.searchsorted(c, positions)

    # -- outputs -------------------------------------------------------------
    def p_exist(self):
        return self.e.mean()

    def estimate(self):
        if not self.e.any():
            return np.full(5, np.nan)
        return self.x[self.e].mean(axis=0)

    def step(self, z):
        self.predict()
        self.update(z)
        return self.p_exist(), self.estimate()


# ----------------------------------------------------------------------------
# Run, score and plot
# ----------------------------------------------------------------------------
def run(snr_db, seed, plot_path):
    rng = np.random.default_rng(seed)
    frames, truth, present = make_sequence(snr_db, rng)
    pf = SalmondBirchPF(N_PARTICLES, rng)

    p_hist = np.zeros(N_FRAMES)
    est = np.full((N_FRAMES, 5), np.nan)
    t0 = time.perf_counter()
    for k in range(N_FRAMES):
        p_hist[k], est[k] = pf.step(frames[k])
    dt = (time.perf_counter() - t0) / N_FRAMES

    declared = p_hist > DETECT_THRESHOLD
    on = np.flatnonzero(declared & present)
    first_det = on[0] if on.size else None
    false_before = int(np.sum(declared[:T_APPEAR]))
    after = declared[T_VANISH:]
    drop_lag = int(np.argmin(after)) if not after.all() else after.size
    false_late = int(np.sum(after[drop_lag:]))
    track = declared & present
    err = np.hypot(est[:, 0] - truth[:, 0], est[:, 2] - truth[:, 2])
    rmse = np.sqrt(np.nanmean(err[track] ** 2)) if track.any() else np.nan

    # single-frame baseline: would the brightest pixel give the target away?
    target_peak = np.array([frames[k][int(round(truth[k, 2])), int(round(truth[k, 0]))]
                            for k in range(N_FRAMES) if present[k]])
    frame_max = frames.reshape(N_FRAMES, -1).max(axis=1)

    print(f"Peak SNR {snr_db:.1f} dB (target amplitude "
          f"{NOISE_STD * 10 ** (snr_db / 20):.2f} x noise std), seed {seed}")
    print(f"Particles {N_PARTICLES}, {dt * 1e3:.1f} ms per {IMG_SIZE}x{IMG_SIZE} frame")
    print(f"Target present frames {T_APPEAR}-{T_VANISH - 1}")
    if first_det is not None:
        print(f"First declared at frame {first_det} "
              f"({first_det - T_APPEAR} frames after appearing)")
    else:
        print("Target never declared")
    print(f"False alarms before the target appeared: {false_before} frames")
    print(f"Track dropped {drop_lag} frames after the target vanished, "
          f"then {false_late} false-alarm frames")
    print(f"Frames tracked: {track.sum()} of {present.sum()}, "
          f"position RMSE {rmse:.2f} px")
    print(f"Single-frame check: the pixel at the target was the brightest in "
          f"the image in {np.sum(target_peak >= frame_max[present])} of "
          f"{present.sum()} frames")

    if plot_path:
        plot(frames, truth, present, p_hist, est, err, declared, snr_db, plot_path)
    return p_hist, est


def plot(frames, truth, present, p_hist, est, err, declared, snr_db, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # show the frame where the target pixel is faintest (a typical hard frame)
    ks = np.flatnonzero(present)
    vals = [frames[k][int(round(truth[k, 2])), int(round(truth[k, 0]))] for k in ks]
    k_show = ks[int(np.argmin(vals))]
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    a = ax[0, 0]
    a.imshow(frames[k_show], cmap="gray", origin="upper")
    a.add_patch(plt.Circle((truth[k_show, 0], truth[k_show, 2]), 3,
                           fill=False, color="tab:red", lw=1.5))
    a.set_title(f"Raw frame {k_show} (target inside red circle)")
    a.set_xticks([]); a.set_yticks([])

    a = ax[0, 1]
    t = np.arange(N_FRAMES)
    a.axvspan(T_APPEAR - 0.5, T_VANISH - 0.5, color="tab:green", alpha=0.12,
              label="target present")
    a.plot(t, p_hist, color="tab:blue", lw=2, label="P(target present)")
    a.axhline(DETECT_THRESHOLD, color="gray", ls="--", lw=1, label="threshold")
    a.set_ylim(-0.02, 1.02)
    a.set_xlabel("frame"); a.set_ylabel("probability")
    a.set_title("Existence probability")
    a.legend(loc="upper left", fontsize=9)

    a = ax[1, 0]
    a.imshow(frames[present].mean(axis=0), cmap="gray", origin="upper")
    a.plot(truth[present, 0], truth[present, 2], color="tab:green", lw=2,
           label="truth")
    m = declared
    a.plot(est[m, 0], est[m, 2], ".", color="tab:orange", ms=7,
           label="PF estimate (declared frames)")
    a.set_xlim(0, IMG_SIZE - 1); a.set_ylim(IMG_SIZE - 1, 0)
    a.set_title("Trajectory (over mean of target frames)")
    a.legend(loc="lower right", fontsize=9)
    a.set_xticks([]); a.set_yticks([])

    a = ax[1, 1]
    a.axvspan(T_APPEAR - 0.5, T_VANISH - 0.5, color="tab:green", alpha=0.12)
    e = np.where(declared & present, err, np.nan)
    a.plot(t, e, "o-", color="tab:orange", ms=4)
    a.set_xlabel("frame"); a.set_ylabel("position error (px)")
    a.set_xlim(0, N_FRAMES - 1)
    a.set_title("Position error while declared")

    fig.suptitle(f"Salmond-Birch PF-TBD, peak SNR {snr_db:.1f} dB, "
                 f"{N_PARTICLES} particles", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Plot written to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snr", type=float, default=6.0, help="peak SNR in dB")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plot", default="pf_tbd_results.png",
                    help="output figure path ('' to skip)")
    args = ap.parse_args()
    run(args.snr, args.seed, args.plot)

# ----------------------------------------------------------------------------
# Notes for real drone imagery
# ----------------------------------------------------------------------------
# 1. Whitening: feed the filter a background-suppressed residual (registered
#    frame minus background estimate, or a local-contrast map) divided by a
#    local noise std estimate, so NOISE_STD = 1 holds approximately.
# 2. Full-frame search with a uniform birth density is the expensive part:
#    most particles are spent looking for a target that is not there. In
#    practice, seed births around low-threshold candidates (a measurement-
#    driven birth density) and run one small filter per region of interest.
# 3. Intensity flicker: replace the random walk on I with a fluctuating-
#    amplitude likelihood (Rutten, Gordon & Maskell, IEE Proc. RSN 2005).
# 4. Tune P_BIRTH / P_DEATH and DETECT_THRESHOLD against false alarms per hour
#    on recorded empty sky, not on synthetic data.
