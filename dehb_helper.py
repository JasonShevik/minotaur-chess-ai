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

        self._best_fitness = np.inf
        self._best_metric_raw: Optional[float] = None
        self._best_config: Optional[Dict[str, Any]] = None

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

    def _maybe_update_best(self, *, config: Dict[str, Any], fitness: float, metric_raw: float) -> None:
        if fitness < self._best_fitness:
            self._best_fitness = float(fitness)
            self._best_metric_raw = float(metric_raw)
            self._best_config = _json_safe(dict(config))

            payload = {
                "best_config": self._best_config,
                "best_fitness": self._best_fitness,
                "best_metric_raw": self._best_metric_raw,
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
        self._maybe_update_best(config=config_dict, fitness=fitness, metric_raw=metric_raw)

        return {"fitness": fitness, "cost": cost, "info": info}