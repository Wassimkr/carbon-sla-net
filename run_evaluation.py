"""Master evaluation script — produces all tables and figures for Carbon-SLA-Net.

Run:
    python run_evaluation.py                         # full evaluation (~4 h)
    python run_evaluation.py --fast                  # 10 episodes, quick check
    python run_evaluation.py --figures-only          # regenerate figures from CSVs
    python run_evaluation.py --skip-milp             # skip slow MILP evaluation
    python run_evaluation.py --model-path <path>     # specify trained model
"""

from __future__ import annotations

_FIG_DIR  = "results/figures"
_SAVE_DIR = "results/tables"
_MODEL_PATH_DEFAULT = "checkpoints/ppo/carbon_sla_net_best/best_model.zip"


def regenerate_all_figures(
    output_dir: str = _FIG_DIR,
    model_path: str = _MODEL_PATH_DEFAULT,
) -> dict[str, str]:
    """Run all 7 figure functions in order, catching per-figure exceptions.

    Parameters
    ----------
    output_dir:
        Directory where PNG/PDF files are saved.
    model_path:
        Path to trained model (used for Figure 8 battery dynamics).

    Returns
    -------
    Dict mapping figure key to PNG path (only for figures that succeeded).
    """
    from figures.figure2_training import plot_training_curves
    from figures.figure3_scalability import plot_scalability
    from figures.figure4_pareto import plot_pareto_front
    from figures.figure5_renewable import plot_renewable_sensitivity
    from figures.figure6_inference import plot_inference_crossover
    from figures.figure7_generalization import plot_generalization_heatmap
    from figures.figure8_battery import plot_battery_dynamics

    tasks = [
        ("fig2", "Figure 2 — Training curves",
         lambda: plot_training_curves(output_dir=output_dir)),
        ("fig3", "Figure 3 — Scalability",
         lambda: plot_scalability(output_dir=output_dir)),
        ("fig4", "Figure 4 — Pareto front",
         lambda: plot_pareto_front(output_dir=output_dir)),
        ("fig5", "Figure 5 — Renewable sensitivity",
         lambda: plot_renewable_sensitivity(output_dir=output_dir)),
        ("fig6", "Figure 6 — Inference crossover",
         lambda: plot_inference_crossover(output_dir=output_dir)),
        ("fig7", "Figure 7 — Generalization heatmap",
         lambda: plot_generalization_heatmap(output_dir=output_dir)),
        ("fig8", "Figure 8 — Battery dynamics",
         lambda: plot_battery_dynamics(model_path=model_path, output_dir=output_dir)),
    ]

    succeeded: dict[str, str] = {}
    failed: list[str] = []

    for key, title, fn in tasks:
        print(f"  {title} …", end=" ", flush=True)
        try:
            path = fn()
            succeeded[key] = path
            print("OK")
        except Exception as exc:
            print(f"FAILED — {type(exc).__name__}: {exc}")
            failed.append(f"{key}: {exc}")

    print(f"\n  Figures: {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        for msg in failed:
            print(f"    ✗ {msg}")
    return succeeded


if __name__ == "__main__":
    import argparse
    import json
    import sys
    import time
    import traceback
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Carbon-SLA-Net full evaluation")
    parser.add_argument(
        "--model-path",
        default="checkpoints/ppo/carbon_sla_net_best/best_model.zip",
    )
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--fast", action="store_true",
                        help="n_episodes=10, n_trials=2 for quick check")
    parser.add_argument("--figures-only", action="store_true",
                        help="Skip evaluation; regenerate figures from existing CSVs")
    parser.add_argument("--skip-milp", action="store_true",
                        help="Skip MILP evaluation (slow at N=100, 200 episodes)")
    args = parser.parse_args()

    if args.fast:
        args.n_episodes = 10

    SAVE_DIR = "results/tables"
    FIG_DIR  = "results/figures"
    ERR_LOG  = "results/evaluation_errors.log"
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    results: dict = {}
    t_pipeline_start = time.perf_counter()

    def log_error(step: str, exc: Exception) -> None:
        msg = f"[{step}] {type(exc).__name__}: {exc}"
        print(f"  WARNING: {msg}")
        errors.append(msg)
        with open(ERR_LOG, "a") as fh:
            fh.write(msg + "\n" + traceback.format_exc() + "\n")

    # ========================================================================
    # Step 1 — Load trained model path
    # ========================================================================
    print(f"\n{'='*60}")
    print("  Carbon-SLA-Net — Evaluation Pipeline")
    print(f"{'='*60}")
    model_path = args.model_path
    model_exists = Path(model_path).exists()
    print(f"  Model path   : {model_path} ({'OK' if model_exists else 'MISSING'})")
    print(f"  n_episodes   : {args.n_episodes}")
    print(f"  figures-only : {args.figures_only}")
    print()

    if not args.figures_only:
        from evaluation.evaluator import Evaluator
        from evaluation.scalability import ScalabilityEvaluator
        from evaluation.ablation import AblationEvaluator
        from evaluation.generalization import GeneralizationEvaluator
        from evaluation.metrics import AggregatedMetrics, aggregate_metrics

        # ====================================================================
        # Step 2 — Main evaluation (all methods)
        # ====================================================================
        print("[Step 2] Main evaluation (all methods) …")
        t0 = time.perf_counter()
        try:
            evaluator = Evaluator(
                N=100, T=8, n_episodes=args.n_episodes,
                seed_offset=10_000,
            )
            milp_path = model_path if model_exists else None
            main_results = evaluator.run_all(
                drl_model_path=milp_path,
                milp_time_limit=60.0 if not args.skip_milp else 0.0,
                save_dir=SAVE_DIR,
                fast_mode=args.fast,
            )
            results["main"] = main_results
        except Exception as e:
            log_error("Step2-main-eval", e)
            main_results = {}
        print(f"  done in {time.perf_counter() - t0:.1f}s")

        # ====================================================================
        # Step 3 — Scalability sweep
        # ====================================================================
        print("[Step 3] Scalability sweep …")
        t0 = time.perf_counter()
        try:
            n_trials = 2 if args.fast else 5
            sc_ev = ScalabilityEvaluator(
                N_values=[10, 20] if args.fast else [50, 100, 150, 200, 300, 400, 500],
                T=8, n_trials=n_trials, seed_offset=50_000,
            )
            sc_results = sc_ev.run(
                drl_model_path=model_path if model_exists else None,
                save_dir=SAVE_DIR, fast_mode=True,
            )
            results["scalability"] = sc_results
        except Exception as e:
            log_error("Step3-scalability", e)
            sc_results = {}
        print(f"  done in {time.perf_counter() - t0:.1f}s")

        # ====================================================================
        # Step 4 — Ablation
        # ====================================================================
        print("[Step 4] Ablation evaluation …")
        t0 = time.perf_counter()
        try:
            if model_exists:
                abl_ev = AblationEvaluator(N=100, T=8,
                                           n_episodes=args.n_episodes,
                                           seed_offset=30_000)
                model_paths = {"full": model_path}
                # Add Pareto-variant models if they exist
                for i in range(7):
                    variant_path = f"checkpoints/ppo/carbon_sla_net_pareto_{i}_best/best_model.zip"
                    if Path(variant_path).exists():
                        model_paths[f"pareto_{i}"] = variant_path
                abl_results = abl_ev.run(model_paths, save_dir=SAVE_DIR)
                results["ablation"] = abl_results
            else:
                print("  Skipped — no trained model")
        except Exception as e:
            log_error("Step4-ablation", e)
        print(f"  done in {time.perf_counter() - t0:.1f}s")

        # ====================================================================
        # Step 5 — Generalization
        # ====================================================================
        print("[Step 5] Generalization evaluation …")
        t0 = time.perf_counter()
        try:
            if model_exists:
                gen_ev = GeneralizationEvaluator(
                    N=100, T=8, n_episodes=args.n_episodes, seed_offset=40_000
                )
                gen_results = gen_ev.run(model_path, save_dir=SAVE_DIR)
                results["generalization"] = gen_results
            else:
                print("  Skipped — no trained model")
                gen_results = {}
        except Exception as e:
            log_error("Step5-generalization", e)
            gen_results = {}
        print(f"  done in {time.perf_counter() - t0:.1f}s")

        # ====================================================================
        # Step 6 — RE sensitivity (3 conditions × same methods)
        # ====================================================================
        print("[Step 6] Renewable sensitivity evaluation …")
        t0 = time.perf_counter()
        re_results: dict = {}
        try:
            from evaluation.evaluator import Evaluator as _Eval
            for condition in ["no_re", "cloudy", "sunny"]:
                ev_cond = _Eval(
                    N=100, T=8, n_episodes=args.n_episodes,
                    seed_offset=10_000, renewable_condition=condition,
                )
                re_results[condition] = ev_cond.run_all(
                    drl_model_path=model_path if model_exists else None,
                    milp_time_limit=30.0,
                    save_dir=SAVE_DIR, fast_mode=True,
                )
            results["re_sensitivity"] = re_results
        except Exception as e:
            log_error("Step6-re-sensitivity", e)
        print(f"  done in {time.perf_counter() - t0:.1f}s")

    # ========================================================================
    # Steps 7–13: Figures
    # ========================================================================
    from figures.figure2_training import plot_training_curves
    from figures.figure3_scalability import plot_scalability
    from figures.figure4_pareto import plot_pareto_front
    from figures.figure5_renewable import plot_renewable_sensitivity
    from figures.figure6_inference import plot_inference_crossover
    from figures.figure7_generalization import plot_generalization_heatmap
    from figures.figure8_battery import plot_battery_dynamics

    fig_paths: dict[str, str] = {}

    print("\n[Step 7] Figure 2 — Training curves …")
    try:
        fig_paths["fig2"] = plot_training_curves(output_dir=FIG_DIR)
    except Exception as e:
        log_error("Fig2", e)

    print("[Step 8] Figure 3 — Scalability …")
    try:
        fig_paths["fig3"] = plot_scalability(output_dir=FIG_DIR)
    except Exception as e:
        log_error("Fig3", e)

    print("[Step 9] Figure 4 — Pareto front …")
    try:
        fig_paths["fig4"] = plot_pareto_front(output_dir=FIG_DIR)
    except Exception as e:
        log_error("Fig4", e)

    print("[Step 10] Figure 5 — Renewable sensitivity …")
    try:
        re_data = results.get("re_sensitivity", {})
        if re_data:
            fig_paths["fig5"] = plot_renewable_sensitivity(re_data, output_dir=FIG_DIR)
        else:
            print("  Skipped — no RE sensitivity results")
    except Exception as e:
        log_error("Fig5", e)

    print("[Step 11] Figure 6 — Inference crossover …")
    try:
        fig_paths["fig6"] = plot_inference_crossover(output_dir=FIG_DIR)
    except Exception as e:
        log_error("Fig6", e)

    print("[Step 12] Figure 7 — Generalization heatmap …")
    try:
        fig_paths["fig7"] = plot_generalization_heatmap(output_dir=FIG_DIR)
    except Exception as e:
        log_error("Fig7", e)

    print("[Step 13] Figure 8 — Battery dynamics …")
    try:
        fig_paths["fig8"] = plot_battery_dynamics(
            model_path=model_path, output_dir=FIG_DIR
        )
    except Exception as e:
        log_error("Fig8", e)

    # ========================================================================
    # Step 14: Print final summary table
    # ========================================================================
    total_min = (time.perf_counter() - t_pipeline_start) / 60.0

    def _fmt(results_dict, method, col):
        agg = results_dict.get(method)
        if agg is None:
            return "—"
        val = getattr(agg, col, None)
        return f"{val:.1f}" if val is not None else "—"

    mr = results.get("main", {})
    print(f"\n{'═'*60}")
    print("  Carbon-SLA-Net — Evaluation Complete")
    print(f"{'═'*60}")
    print("  Main results (N=100, sunny):")
    print(f"  {'Method':<18} {'E (Wh)':>9} {'CF(gCO2)':>9} {'SLA viol':>9}")
    print(f"  {'─'*49}")
    for method in ["MILP", "Carbon-SLA-Net", "NSGA-II", "RenewableGreedy",
                   "EnergyGreedy", "SLAPriority"]:
        if method in mr:
            agg = mr[method]
            print(f"  {method:<18} {agg.mean_energy_wh:>9.1f} "
                  f"{agg.mean_carbon_gco2:>9.1f} {agg.mean_sla_violation:>9.1f}")
    print(f"  Figures saved to: {FIG_DIR}/")
    print(f"  Tables saved to:  {SAVE_DIR}/")
    print(f"  Total time: {total_min:.1f} min")
    if errors:
        print(f"  Errors ({len(errors)}): see {ERR_LOG}")
    print(f"{'═'*60}")

    # ========================================================================
    # Step 15: Save JSON summary
    # ========================================================================
    summary = {
        "model_path": model_path,
        "n_episodes": args.n_episodes,
        "total_duration_min": round(total_min, 2),
        "figures_produced": list(fig_paths.keys()),
        "errors": errors,
    }
    summary_path = "results/run_evaluation_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Summary JSON: {summary_path}")
