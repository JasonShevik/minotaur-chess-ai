from __future__ import annotations

import dataclasses
import json
import os
import time
import numpy as np
import ConfigSpace as CS
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Literal, Tuple, Union
from dehb import DEHB


Direction = Literal["minimize", "maximize"]


@dataclass
class ObjectiveResult:
    """
    What your objective returns.

    - metric: the primary scalar you care about (e.g., validation loss OR similarity score)
    - info: anything else you want logged (must be JSON-serializable if you want clean logs)
    """
    metric: float
    info: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass
class DEHBRunConfig:
    # Fidelity range: DEHB uses min_fidelity/max_fidelity
    min_fidelity: float
    max_fidelity: float
    eta: int = 3  # hyperband downsampling rate

    # How long to run (choose ONE typically)
    fevals: Optional[int] = None         # number of function evals
    brackets: Optional[int] = None       # number of SH brackets
    total_cost: Optional[int] = None     # wallclock seconds aggregated across evals

    # Parallelism (DEHB can manage dask internally, but you can keep it 1 for now)
    n_workers: int = 1
    single_node_with_gpus: bool = False  # DEHB can assign workers to GPUs when using run()

    # Reproducibility
    seed: int = 0

    # DE knobs (optional)
    mutation_factor: Optional[float] = None
    crossover_prob: Optional[float] = None
    strategy: Optional[str] = None  # DE strategy string (leave None for defaults)

    # Logging / saving
    output_path: Union[str, Path] = "dehb_out"
    log_level: str = "INFO"  # loguru levels used by DEHB
    resume: bool = False     # restart from saved state if output_path has it
    save_freq: Optional[str] = None  # when DEHB writes its state: "step" (after every evaluation), "incumbent"
                                     # (its default: when the best changes) or "end". A search that may be
                                     # stopped at any time wants "step"

    # Which evaluation counts as the best. False: the best metric of all, whatever its fidelity. True: the
    # best among those at the highest fidelity evaluated so far (a short run and a long one do not compare)
    best_at_highest_fidelity: bool = False


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy / torch scalar types to plain Python so
    json.dumps doesn't choke on ConfigSpace's np.int64 / np.float64 values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


class DEHBHelper:
    """
    A reusable wrapper around DEHB that:
      - enforces a stable objective signature
      - handles maximize/minimize
      - logs evaluations
      - saves the best config
    """

    def __init__(
        self,
        *,
        configspace: CS.ConfigurationSpace,
        objective: Callable[[Dict[str, Any], float], ObjectiveResult],
        direction: Direction,
        run_cfg: DEHBRunConfig,
        # Optional: attach arbitrary static context that your objective can read
        static_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cs = configspace
        self.objective = objective
        self.direction = direction
        self.run_cfg = run_cfg
        self.static_context = static_context or {}

        self.output_path = Path(run_cfg.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self._jsonl_path = self.output_path / "evals.jsonl"
        self._best_path = self.output_path / "best_config.json"
        self._pending_path = self.output_path / "pending_job.json"

        self._best_fitness = np.inf
        self._best_metric_raw: Optional[float] = None
        self._best_config: Optional[Dict[str, Any]] = None
        self._best_fidelity: float = -np.inf

        # On resume, reload our own incumbent tracking from disk so that
        # best_config.json stays trustworthy mid-run instead of only being
        # rewritten once a post-resume trial beats a fresh inf baseline.
        if run_cfg.resume and self._best_path.exists():
            try:
                with self._best_path.open("r", encoding="utf-8") as f:
                    prev = json.load(f)
                if prev.get("direction") not in (None, self.direction):
                    raise ValueError(
                        f"Saved incumbent used direction={prev.get('direction')!r}, "
                        f"but this run uses direction={self.direction!r}."
                    )
                self._best_config = prev.get("best_config")
                self._best_fitness = float(prev.get("best_fitness", np.inf))
                self._best_metric_raw = prev.get("best_metric_raw")
                self._best_fidelity = float(prev.get("best_fidelity", -np.inf))
                print(
                    f"[DEHBHelper] Resumed incumbent from {self._best_path} "
                    f"(best_metric_raw={self._best_metric_raw})"
                )
            except ValueError:
                raise
            except Exception as e:
                print(f"[DEHBHelper] Could not restore incumbent ({e}); tracking fresh.")

        # Initialize DEHB (DEHB expects a function f(config, fidelity, **kwargs) for run())
        # We wrap your objective to match DEHB's expected result dict: {fitness, cost, info}.
        # NOTE: only pass DE knobs when explicitly set — passing None would
        # override DEHB's internal defaults (0.5 / 0.5 / 'rand1_bin') with None
        # and break the DE mutation/crossover steps.
        de_kwargs: Dict[str, Any] = {}
        if self.run_cfg.mutation_factor is not None:
            de_kwargs["mutation_factor"] = self.run_cfg.mutation_factor
        if self.run_cfg.crossover_prob is not None:
            de_kwargs["crossover_prob"] = self.run_cfg.crossover_prob
        if self.run_cfg.strategy is not None:
            de_kwargs["strategy"] = self.run_cfg.strategy
        if self.run_cfg.save_freq is not None:
            de_kwargs["save_freq"] = self.run_cfg.save_freq

        self._dehb = DEHB(
            cs=self.cs,
            f=self._dehb_objective_wrapper,
            min_fidelity=self.run_cfg.min_fidelity,
            max_fidelity=self.run_cfg.max_fidelity,
            eta=self.run_cfg.eta,
            n_workers=self.run_cfg.n_workers,
            seed=self.run_cfg.seed,
            output_path=str(self.output_path),
            log_level=self.run_cfg.log_level,
            resume=self.run_cfg.resume,
            **de_kwargs,
        )

    # ---------- Public API ----------

    def run(self) -> Dict[str, Any]:
        """
        Runs DEHB using its built-in scheduler (and optional dask) via DEHB.run().

        Returns a summary dict with incumbent and history pointers.
        """
        fevals = self.run_cfg.fevals
        brackets = self.run_cfg.brackets
        total_cost = self.run_cfg.total_cost

        if fevals is None and brackets is None and total_cost is None:
            raise ValueError("Set one of run_cfg.fevals, run_cfg.brackets, or run_cfg.total_cost.")

        traj, runtime, history = self._dehb.run(
            fevals=fevals,
            brackets=brackets,
            total_cost=total_cost,
            single_node_with_gpus=self.run_cfg.single_node_with_gpus,
            # Any additional kwargs here get broadcast to workers in DEHB when using dask.
        )

        # DEHB keeps its own incumbent internally; but we also maintained our own best.
        summary = {
            "output_path": str(self.output_path),
            "best_config": self._best_config,
            "best_fitness": float(self._best_fitness),
            "best_metric_raw": None if self._best_metric_raw is None else float(self._best_metric_raw),
            "trajectory_len": len(traj),
            "history_len": len(history),
            "eval_log": str(self._jsonl_path),
            "best_config_path": str(self._best_path),
        }
        return summary

    def run_stoppable(
        self,
        should_stop: Callable[[], bool],
        *,
        fevals: Optional[int] = None,
        seconds: Optional[float] = None,
        interrupted: Union[type, Tuple[type, ...]] = (),
    ) -> Dict[str, Any]:
        """
        Runs DEHB one evaluation at a time (one worker), so the search can be stopped and taken up again.

        should_stop: asked before every evaluation; True ends the search. An objective that is stopped part-way
            says so by raising one of `interrupted`: nothing of that evaluation is reported, the job stays
            written in pending_job.json, and the next start (a new helper, resume=True) gives the objective the
            same configuration and fidelity first, so an objective that kept a checkpoint can carry on with it.
        fevals: stop once this many evaluations are done in all, earlier starts included.
        seconds: stop asking for new evaluations after this long (the one under way finishes).

        Use save_freq="step" in the run config, so DEHB's own state is on disk after every evaluation.
        After this returns, do not keep using the helper: build a new one to continue.
        """
        started = time.time()
        pending = self._read_pending()
        reason = "stopped"
        while True:
            if should_stop():
                break
            if fevals is not None and len(self._dehb.traj) >= fevals:
                reason = "evaluations"
                break
            if seconds is not None and time.time() - started >= seconds:
                reason = "time"
                break
            job = self._dehb.ask()
            if pending is not None:
                self._adopt(job, pending)
                pending = None
            self._write_pending(job)
            try:
                result = self._dehb_objective_wrapper(job["config"], job["fidelity"])
            except interrupted:
                break                      # pending_job.json stays: the next start takes this evaluation up again
            self._dehb.tell(job, result)
            self._pending_path.unlink(missing_ok=True)
            self._dehb._clean_inactive_brackets()
            print(f"[DEHBHelper] evaluation {len(self._dehb.traj)}: fidelity {float(job['fidelity']):g}, metric "
                  f"{self._fitness_to_metric(result['fitness']):.4f}, {result['cost'] / 60:.1f} min; best so far "
                  f"{self._best_metric_raw:.4f} at fidelity {self._best_fidelity:g}; "
                  f"{(time.time() - started) / 3600:.1f} h since this start", flush=True)
        self._dehb.save()
        return {
            "output_path": str(self.output_path),
            "best_config": self._best_config,
            "best_fitness": float(self._best_fitness),
            "best_metric_raw": None if self._best_metric_raw is None else float(self._best_metric_raw),
            "evaluations": len(self._dehb.traj),
            "ended_because": reason,
            "eval_log": str(self._jsonl_path),
            "best_config_path": str(self._best_path),
        }

    def _write_pending(self, job: dict) -> None:
        """The job about to be evaluated, as DEHB holds it: its point of the unit cube, and its fidelity."""
        vector = self._dehb.config_repository.configs[job["config_id"]].config
        with self._pending_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe({"vector": np.asarray(vector).tolist(), "fidelity": float(job["fidelity"])}), f)

    def _read_pending(self) -> Optional[Dict[str, Any]]:
        if not self._pending_path.exists():
            return None
        try:
            with self._pending_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[DEHBHelper] Could not read {self._pending_path} ({e}); the interrupted evaluation is dropped.")
            return None

    def _adopt(self, job: dict, pending: Dict[str, Any]) -> None:
        """
        Make the job DEHB just handed out the one that was interrupted. After a restart DEHB replays the
        finished evaluations and then proposes a job for the same place in its schedule, but with a freshly
        drawn configuration; putting the interrupted one in its place is what DEHB's own replay does with
        every finished evaluation. If the fidelities differ the schedule is not where it was (the settings
        changed): the interrupted evaluation is dropped.
        """
        if float(job["fidelity"]) != float(pending["fidelity"]):
            print("[DEHBHelper] The interrupted evaluation does not fit the schedule any more; dropping it.")
            return
        vector = np.asarray(pending["vector"], dtype=float)
        self._dehb.config_repository.configs[job["config_id"]].config = vector
        job["config"] = self._dehb.vector_to_configspace(vector) if self._dehb.use_configspace else vector

    def ask(self, n: int = 1) -> Union[dict, list[dict]]:
        """
        Manual scheduling mode: ask DEHB for the next job(s).
        You evaluate them yourself, then call tell().
        """
        return self._dehb.ask(n_configs=n)

    def tell(self, job_info: dict, *, metric: float, cost: float, info: Optional[Dict[str, Any]] = None) -> None:
        """
        Manual scheduling mode: report a result back to DEHB.
        metric is in your natural direction; we convert to fitness.
        """
        fitness = self._metric_to_fitness(metric)
        result = {"fitness": float(fitness), "cost": float(cost), "info": info or {}}
        self._dehb.tell(job_info, result)

    # ---------- Internal helpers ----------

    def _metric_to_fitness(self, metric: float) -> float:
        # DEHB minimizes "fitness". If you want to maximize a score, fitness = -score.
        if self.direction == "minimize":
            return float(metric)
        elif self.direction == "maximize":
            return float(-metric)
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

    def _fitness_to_metric(self, fitness: float) -> float:
        if self.direction == "minimize":
            return float(fitness)
        else:
            return float(-fitness)

    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")

    def _maybe_update_best(self, *, config: Dict[str, Any], fitness: float, metric_raw: float,
                           fidelity: float = 0.0, info: Optional[Dict[str, Any]] = None) -> None:
        if self.run_cfg.best_at_highest_fidelity:
            better = fidelity > self._best_fidelity or (fidelity == self._best_fidelity and fitness < self._best_fitness)
        else:
            better = fitness < self._best_fitness
        if better:
            self._best_fitness = float(fitness)
            self._best_metric_raw = float(metric_raw)
            self._best_config = _json_safe(dict(config))
            self._best_fidelity = float(fidelity)

            payload = {
                "best_config": self._best_config,
                "best_fitness": self._best_fitness,
                "best_metric_raw": self._best_metric_raw,
                "best_fidelity": self._best_fidelity,
                "best_info": info or {},
                "direction": self.direction,
                "updated_at_unix": time.time(),
            }
            with self._best_path.open("w", encoding="utf-8") as f:
                json.dump(_json_safe(payload), f, indent=2)

    def _dehb_objective_wrapper(self, config: Any, fidelity: float, **kwargs) -> Dict[str, Any]:
        """
        This is what DEHB calls.

        DEHB's ask/tell docs require result dict with:
          - fitness (float)
          - cost (float)
          - optional info (serializable)
        """
        # Convert ConfigSpace.Configuration -> plain dict (DEHB may already pass dicts depending on mode)
        if hasattr(config, "get_dictionary"):
            config_dict = config.get_dictionary()
        elif isinstance(config, dict):
            config_dict = config
        else:
            # Last-resort conversion
            config_dict = dict(config)

        # Measure wallclock cost
        t0 = time.time()
        out = self.objective(config_dict, float(fidelity))
        t1 = time.time()

        metric_raw = float(out.metric)
        fitness = float(self._metric_to_fitness(metric_raw))
        cost = float(t1 - t0)

        info = dict(out.info or {})
        # Helpful to always log fidelity and raw metric
        info.setdefault("metric_raw", metric_raw)
        info.setdefault("direction", self.direction)
        info.setdefault("fidelity", float(fidelity))

        # Log every eval (JSONL)
        self._append_jsonl(
            {
                "ts_unix": time.time(),
                "config": config_dict,
                "fidelity": float(fidelity),
                "metric_raw": metric_raw,
                "fitness": fitness,
                "cost_sec": cost,
                "info": info,
            }
        )

        # Track incumbent ourselves too
        self._maybe_update_best(config=config_dict, fitness=fitness, metric_raw=metric_raw,
                                fidelity=float(fidelity), info=info)

        # DEHB keeps `info` in a table it writes with pandas, which takes flat values only: nested ones (kept
        # in evals.jsonl and best_config.json above) are left out of what DEHB itself stores
        flat = {k: v for k, v in _json_safe(info).items() if isinstance(v, (int, float, str, bool)) or v is None}
        return {"fitness": fitness, "cost": cost, "info": flat}