import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import argparse
from src.segmentation import (
    segment_dp,
    segment_dp_2d,
    segment_baseline,
    segment_sliding_window,
    segment_dp_overlap,
    compute_metrics,
    compare,
    generate_full_report,
    suggest_lambda,
    estimate_optimal_k,
    validate_calibration,
)


def print_separator():
    print("─" * 60)


def cmd_segment(args):
    print_separator()
    print(f"Método: {args.method.upper()}")
    print(f"lmin={args.lmin}, lmax={args.lmax}, model={args.model}")
    print_separator()

    if args.method == "dp":
        result = segment_dp(
            args.text, lmin=args.lmin, lmax=args.lmax,
            model=args.model, fixed_cost=args.fixed_cost,
        )
    elif args.method == "baseline":
        result = segment_baseline(args.text, lmax=args.lmax, model=args.model)
    elif args.method == "sliding_window":
        result = segment_sliding_window(
            args.text, lmax=args.lmax, overlap=args.overlap, model=args.model,
        )
    elif args.method == "overlap":
        result = segment_dp_overlap(
            args.text, lmin=args.lmin, lmax=args.lmax,
            model=args.model, fixed_cost=args.fixed_cost,
        )
    elif args.method == "dp2d":
        result = segment_dp_2d(
            args.text, lmin=args.lmin, lmax=args.lmax,
            model=args.model, fixed_cost=args.fixed_cost,
        )
        print(f"k óptimo: {result.num_segments}")
        print(f"Costo total: {result.total_cost:,.0f} tokens²")
        print_separator()
        print("Costo por k:")
        for k, cost in result.cost_by_k.items():
            marker = " ← óptimo" if k == result.num_segments else ""
            print(f"  k={k}: {cost:,.0f}{marker}")
        print_separator()
        for seg in result.segments:
            print(f"\nSegmento {seg.index} — {seg.token_count} tokens")
            print(f"  {seg.text[:100]}{'...' if len(seg.text) > 100 else ''}")
        return

    metrics = compute_metrics(result)
    print(f"Segmentos:   {metrics.num_segments}")
    print(f"Costo total: {metrics.total_cost:,.0f} tokens²")
    print(f"Avg tokens:  {metrics.avg_tokens_per_segment}")
    print(f"Std tokens:  {metrics.std_tokens_per_segment}")
    print(f"Coherencia:  {metrics.avg_coherence}")
    print_separator()

    for seg in result.segments:
        print(f"\nSegmento {seg.index} — {seg.token_count} tokens")
        print(f"  {seg.text[:100]}{'...' if len(seg.text) > 100 else ''}")


def cmd_compare(args):
    print_separator()
    print("Comparación DP vs Baseline vs Sliding Window")
    print(f"lmin={args.lmin}, lmax={args.lmax}, model={args.model}")
    print_separator()

    dp_result = segment_dp(
        args.text, lmin=args.lmin, lmax=args.lmax,
        model=args.model, fixed_cost=args.fixed_cost,
    )
    base_result = segment_baseline(args.text, lmax=args.lmax, model=args.model)
    report = compare(dp_result, base_result)

    print(f"{'Método':<20} {'Segmentos':>10} {'Costo':>12} {'Avg tok':>10} {'Coherencia':>12}")
    print("─" * 70)
    print(f"{'DP Optimal':<20} {report.dp.num_segments:>10} {report.dp.total_cost:>12,.0f} {report.dp.avg_tokens_per_segment:>10} {report.dp.avg_coherence:>12}")
    print(f"{'Baseline':<20} {report.baseline.num_segments:>10} {report.baseline.total_cost:>12,.0f} {report.baseline.avg_tokens_per_segment:>10} {report.baseline.avg_coherence:>12}")
    print_separator()
    print(f"Reducción de costo:     {report.cost_reduction_pct}%")
    print(f"Mejora de coherencia:   {report.coherence_improvement:+.4f}")
    print(f"Diferencia segmentos:   {report.segment_diff}")


def cmd_report(args):
    print_separator()
    print("Generando reporte completo...")
    print_separator()

    report = generate_full_report(
        args.text,
        lmin=args.lmin,
        lmax=args.lmax,
        model=args.model,
        fixed_cost=args.fixed_cost,
        run_llm_evaluation=args.llm,
    )

    print(f"Tokens totales:  {report.text_length_tokens}")
    print(f"Oraciones:       {report.num_sentences}")
    print_separator()
    print(f"{'Método':<20} {'Segmentos':>10} {'Costo':>12}")
    print("─" * 45)
    print(f"{'DP Optimal':<20} {report.dp_metrics['num_segments']:>10} {report.dp_metrics['total_cost']:>12,.0f}")
    print(f"{'Baseline':<20} {report.baseline_metrics['num_segments']:>10} {report.baseline_metrics['total_cost']:>12,.0f}")
    print(f"{'Sliding Window':<20} {report.sliding_window_metrics['num_segments']:>10} {report.sliding_window_metrics['total_cost']:>12,.0f}")
    print(f"{'DP + Overlap':<20} {report.overlap_metrics['num_segments']:>10} {report.overlap_metrics['total_cost']:>12,.0f}")
    print(f"{'DP 2D':<20} {report.dp_2d_metrics['optimal_k']:>10} {report.dp_2d_metrics['total_cost']:>12,.0f}")
    print_separator()
    print(f"Reducción costo DP vs Baseline:        {report.dp_vs_baseline['cost_reduction_pct']}%")
    print(f"Mejora coherencia DP vs Baseline:      {report.dp_vs_baseline['coherence_improvement']:+.4f}")
    print(f"Reducción costo DP vs Sliding Window:  {report.dp_vs_sliding_window['cost_reduction_pct']}%")
    print_separator()
    print(f"k* analítico:        {report.dp_2d_metrics['k_analytical']}")
    print(f"k* DP 2D:            {report.dp_2d_metrics['optimal_k']}")
    print(f"Desviación:          {report.dp_2d_metrics['deviation_pct']:.1%}")
    print(f"Calibración válida:  {'Sí' if report.dp_2d_metrics['calibration_valid'] else 'No'}")


def cmd_calibrate(args):
    print_separator()
    print("Calibración de parámetros")
    print(f"lmin={args.lmin}, lmax={args.lmax}, fixed_cost={args.fixed_cost}")
    print_separator()

    lambda_sugerido = suggest_lambda(args.text)
    print(f"λ sugerido (coherencia):  {lambda_sugerido}")

    k_star = estimate_optimal_k(args.text, model=args.model, fixed_cost=args.fixed_cost)
    print(f"k* analítico:             {k_star}")

    print_separator()
    print("Validando calibración con DP 2D...")
    calibration = validate_calibration(
        args.text,
        lmin=args.lmin,
        lmax=args.lmax,
        coherence_lambda=lambda_sugerido,
        fixed_cost=args.fixed_cost,
        model=args.model,
    )
    print(f"k* DP 2D:                 {calibration['k_dp']}")
    print(f"k* analítico:             {calibration['k_analytical']}")
    print(f"Desviación:               {calibration['deviation_pct']:.1%}")
    print(f"Calibración válida:       {'Sí ✓' if calibration['valid'] else 'No ✗'}")
    print_separator()

    if not calibration["valid"]:
        print("Recomendación: ajusta fixed_cost o lmin/lmax para mejorar la calibración.")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Optimal Segmentation CLI"
    )
    parser.add_argument("--text", type=str, required=True, help="Texto a segmentar")
    parser.add_argument("--lmin", type=int, default=20, help="Tokens mínimos por segmento")
    parser.add_argument("--lmax", type=int, default=150, help="Tokens máximos por segmento")
    parser.add_argument("--overlap", type=int, default=20, help="Overlap para sliding window")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Modelo de tokenización")
    parser.add_argument("--fixed-cost", type=float, default=1000.0, help="Costo fijo por segmento")

    subparsers = parser.add_subparsers(dest="command")

    # comando segment
    seg_parser = subparsers.add_parser("segment", help="Segmentar con un método específico")
    seg_parser.add_argument(
        "--method", type=str, default="dp",
        choices=["dp", "baseline", "sliding_window", "overlap", "dp2d"],
        help="Método de segmentación"
    )

    # comando compare
    subparsers.add_parser("compare", help="Comparar DP vs Baseline")

    # comando report
    rep_parser = subparsers.add_parser("report", help="Reporte completo de todos los métodos")
    rep_parser.add_argument("--llm", action="store_true", help="Incluir evaluación con LLM real")

    # comando calibrate
    cal_parser = subparsers.add_parser("calibrate", help="Calibrar parámetros óptimos para el texto")
    cal_parser.add_argument(
        "--llm-model", type=str, default="gemma2:2b",
        help="Modelo Ollama para calibrar el costo fijo (requiere Ollama corriendo)"
    )

    args = parser.parse_args()

    if args.command == "segment":
        cmd_segment(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()