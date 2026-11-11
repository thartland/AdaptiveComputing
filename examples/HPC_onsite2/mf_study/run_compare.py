"""Compare fidelity-selection policies on Tucker's 6D Hartmann problem at an
equal adaptive-cost budget.

Policies
  promo         promotion-first (new driver), streak guard 5
  promo_noguard promotion-first, streak guard disabled
  alwayshf      HF-EI location, nested LF+HF every step (what the unfixed
                MF-SEGO does at power 0.5 and the cost-ratio driver did here)
  promo_gN      promotion-first with streak guard N (3, 5 or 10)
  sfbo          plain single-fidelity BO on HF only (cost 3 per point),
                initial design = the same HF points, no LF data
  mfkg, mfmes   multifidelity knowledge gradient / max-value entropy search
                (hero_mf_information.py): location and fidelity by value per
                cost, no exponent or streak guard

--rho linear uses SMT_GP(mfk_kwargs={"rho_regr": "linear"}) for the co-kriging
level, with a constant-rho fallback while there are too few HF points.
--n-init-hf sets the initial HF design (default 8); LF gets the same points
plus 2. np.random draws the HF design first, so the first 8 points of a
12-point design are the 8-point design of the same seed.
"""
import argparse, json, os, sys, time
import numpy as np

W = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root
sys.path.insert(0, W)
sys.path.insert(0, os.path.join(W, "examples", "HPC_onsite2"))

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverHeroMFSEGO, ActiveLoopDriverHeroMFPromotion, ActiveLoopDriverHeroMFInfo
from adaptive_computing.local_hero import LocalHeroClient
from local_manager import LocalFunctionManager
from simulation_files.hartmann import hf_hartmann as func_hf, lf_hartmann
from simulation_files.hartmann_mf import lf_close, lf_forrester

Y_TRUE_MIN = -3.32237
ndim = 6

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True, choices=["promo", "promo_noguard", "alwayshf", "promo_g3", "promo_g5", "promo_g10", "sfbo", "mfkg", "mfmes"])
ap.add_argument("--seed", type=int, default=38)
ap.add_argument("--budget", type=float, default=120.0, help="adaptive cost budget")
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--out", required=True)
ap.add_argument("--lf", default="tucker", choices=["tucker", "close", "forrester"])
ap.add_argument("--hf-cost", type=float, default=3.0, help="HF cost; LF cost is 1")
ap.add_argument("--rho", default="constant", choices=["constant", "linear", "quadratic"])
ap.add_argument("--n-init-hf", type=int, default=8)
ap.add_argument("--hyper", default="mle", choices=["mle", "bayes"], help="mfkg/mfmes: discrepancy hyperparameters fitted (mle) or averaged (bayes)")
ap.add_argument("--lf-guard", type=float, default=None, help="mfkg/mfmes: force HF after this much LF cost (HF-equivalents) without HF")
args = ap.parse_args()
func_lf = {"tucker": lf_hartmann, "close": lf_close, "forrester": lf_forrester}[args.lf]


class AlwaysHF(ActiveLoopDriverHeroMFSEGO):
    def get_next_sample(self):
        kb_s, kb_d = self._build_kriging_believer_surrogate()
        x = np.atleast_2d(self.sampler.minimize_acq_func(kb_s, kb_d, i_fidelity=self.n_fidelity - 1))
        self.last_candidate_source = "new"
        self.last_mfsego_scores = np.full(self.n_fidelity, np.nan)
        self.last_mfsego_eligible = np.ones(self.n_fidelity, dtype=bool)
        return x, self.n_fidelity - 1


db = args.out + ".db.json"
for f in [db, db + ".lock"]:
    if os.path.exists(f):
        os.remove(f)
local_hero = LocalHeroClient(db_path=db)
sf = args.policy == "sfbo"
manager = LocalFunctionManager(hero_client=local_hero,
                               simulation_functions=[func_hf] if sf else [func_lf, func_hf],
                               machine_name="local", max_workers=4)
params = [ContinuousVariable(min=0, max=1)] * ndim
common = dict(simulations=[None] if sf else [None, None], fidelity_costs=[args.hf_cost] if sf else [1., args.hf_cost], params=params,
              machine_names=[manager.machine_name], output_field_path='y_data', surrogate='SMT_GP',
              acq_func='expected_improvement', blocking=False, inline_manager=manager,
              hero_client=local_hero, reference_minimum=Y_TRUE_MIN)
if args.policy == "promo":
    drv = ActiveLoopDriverHeroMFPromotion(**common)
elif args.policy == "promo_noguard":
    drv = ActiveLoopDriverHeroMFPromotion(max_low_fidelity_streak=None, **common)
elif args.policy in ("mfkg", "mfmes"):
    drv = ActiveLoopDriverHeroMFInfo(acquisition=args.policy[2:], seed=args.seed,
                                     max_lf_cost_without_hf=args.lf_guard, hyper=args.hyper, **common)
elif args.policy.startswith("promo_g"):
    drv = ActiveLoopDriverHeroMFPromotion(max_low_fidelity_streak=int(args.policy[len("promo_g"):]), **common)
else:  # alwayshf, or sfbo where the only (highest) level is HF
    drv = AlwaysHF(mfsego_cost_power=1.0, promotion_ei_fraction=None, **common)

if args.rho != "constant" and not sf:
    from adaptive_computing.surrogates.smt_gp import SMT_GP
    drv.surrogate = SMT_GP(drv.dataset, mfk_kwargs={"rho_regr": args.rho})
rho_log = []

drv.sampler._rand_seed = args.seed
drv.init_sampler._rand_seed = args.seed
np.random.seed(args.seed)
xHF = np.atleast_2d(np.random.rand(args.n_init_hf, ndim))
xLF = np.vstack([xHF, np.random.rand(2, ndim)])
if sf:
    drv.add_samples(xHF, i_fidelity=0)
    manager.run_until_done(i_fidelities=[0])
else:
    drv.add_samples(xLF, i_fidelity=0)
    drv.add_samples(xHF, i_fidelity=1)
    manager.run_until_done(i_fidelities=[0, 1])
drv.hero_wait_for_data_and_train()


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o


def dump(done):
    out = {"args": vars(args), "elapsed": time.time() - t0, "done": done,
           "cost_regret": clean(drv.total_cost_regret_pairs),
           "adaptive_cost": drv.cumulative_adaptive_cost,
           "selections": clean(drv.selection_history),
           "rho_used": rho_log,
           "y_hf": clean(drv.dataset._y_data[-1]), "y_lf": [] if sf else clean(drv.dataset._y_data[0])}
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, args.out if done else args.out + ".partial")


t0 = time.time()
while drv.cumulative_adaptive_cost < args.budget:
    drv.run(N_steps=args.batch, batch_size=args.batch)
    rho_log.append(getattr(drv.surrogate, "rho_used", [None])[-1])
    dump(False)
    print(f"[progress] policy={args.policy} seed={args.seed} adaptive_cost={drv.cumulative_adaptive_cost:.0f} "
          f"regret={drv.cost_regret_pairs[-1][1]:.4f} rho={rho_log[-1]} elapsed={time.time()-t0:.0f}s", flush=True)
dump(True)
fids = "".join(str(s["selected_fidelity"]) for s in drv.selection_history)
print(f"DONE policy={args.policy} seed={args.seed} cost={drv.cumulative_adaptive_cost:.0f} "
      f"regret={drv.cost_regret_pairs[-1][1]:.4f} fids={fids} elapsed={time.time()-t0:.0f}s")
