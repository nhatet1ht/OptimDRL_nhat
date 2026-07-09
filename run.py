"""
Main script to run the Handover Optimization Framework.

This script provides a unified command-line interface to:
- Train a PPO-based handover policy.
- Validate handover performance using a PPO policy.
- Benchmark using a standard 3GPP-compliant handover algorithm.

Usage:
    python main.py <script>

Arguments:
    script: One of 'train_ppo', 'validate_ppo', or 'validate_3gpp'.
"""

import argparse
import os
import sys
from scripts import plot_results, train_ppo, validate_3gpp, validate_ppo
from scripts import train_ppo_2phase, evaluate_all
from scripts import train_ppo_curriculum, train_ppo_reward_shaped
from scripts import train_ppo_imitation, train_ppo_multiseed
from scripts import train_ppo_bc_curriculum, eval_checkpoints

THIS_PATH = os.path.dirname(os.path.abspath(__file__))


def run() -> int:
    """Run the Handover Optimization Framework."""
    parser = argparse.ArgumentParser(
        description="Handover Optimization Framework\n\n"
        "Use this entry point to train and evaluate different handover strategies:\n"
        "  • 3GPP-standard handover\n"
        "  • PPO-based handover\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="\nAvailable commands:")

    # plot_results
    subparsers.add_parser(
        "plot_results",
        help="Plot the results shown in the paper.",
        description="Runs the script that creates the same plots that are shown in the paper.",
    )

    # train_ppo
    subparsers.add_parser(
        "train_ppo",
        help="Train a PPO policy for handover decisions",
        description="Trains a PPO policy to make optimal handover decisions.",
    )

    # validate_3gpp
    subparsers.add_parser(
        "validate_3gpp",
        help="Validate using 3GPP-compliant handover",
        description="Runs the 3GPP-standard handover validation procedure.",
    )

    # validate_ppo
    subparsers.add_parser(
        "validate_ppo",
        help="Validate using a trained PPO policy",
        description="Runs validation using a pre-trained PPO handover policy.",
    )

    # train_ppo_2phase
    subparsers.add_parser(
        "train_ppo_2phase",
        help="Train PPO with 2-phase curriculum (modified t_ho_prep)",
        description="Phase 1: t_ho_prep=3 steps, no PP penalty. Phase 2: t_ho_prep=5 steps, full penalty.",
    )

    # evaluate_all
    subparsers.add_parser(
        "evaluate_all",
        help="Compare original PPO vs 2-phase PPO across all speeds",
        description="Evaluates both models on 30/50/70/90 km/h datasets and prints comparison.",
    )

    # train_ppo_curriculum
    subparsers.add_parser(
        "train_ppo_curriculum",
        help="3-phase curriculum: t_ho_prep = 1 -> 3 -> 5 (solves credit assignment)",
        description="Phase 0: instant HO. Phase 1: medium delay. Phase 2: full paper params.",
    )

    # train_ppo_reward_shaped
    subparsers.add_parser(
        "train_ppo_reward_shaped",
        help="2-phase PPO with action-based reward shaping (immediate gradient signal)",
        description="Adds r_shaped = 0.1*sinr_norm[action] to provide gradient before HO completes.",
    )

    # train_ppo_imitation
    subparsers.add_parser(
        "train_ppo_imitation",
        help="Behavioral cloning from oracle + PPO fine-tuning",
        description="Stage 1: pre-train actor to pick best SINR BS. Stage 2: PPO 2-phase.",
    )

    # train_ppo_multiseed
    subparsers.add_parser(
        "train_ppo_multiseed",
        help="5 random seeds (2-phase each), best model saved",
        description="Probes whether convergence is seed-sensitive. Seeds: 0,42,123,777,1234.",
    )

    # train_ppo_bc_curriculum
    subparsers.add_parser(
        "train_ppo_bc_curriculum",
        help="BC init + gradual curriculum t_ho_prep=1->5, t_ho_exec=1->4",
        description="Most promising approach: BC gives best-BS init, 5 gradual phases.",
    )

    # eval_checkpoints
    subparsers.add_parser(
        "eval_checkpoints",
        help="Evaluate intermediate checkpoints with their own training config",
        description="Diagnostic: shows if curriculum Phase 0 actually learned good HO behavior.",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    if args.command == "plot_results":
        return plot_results.main(THIS_PATH)
    if args.command == "train_ppo":
        return train_ppo.main(THIS_PATH)
    if args.command == "validate_3gpp":
        return validate_3gpp.main(THIS_PATH)
    if args.command == "validate_ppo":
        return validate_ppo.main(THIS_PATH)
    if args.command == "train_ppo_2phase":
        return train_ppo_2phase.main()
    if args.command == "evaluate_all":
        return evaluate_all.main()
    if args.command == "train_ppo_curriculum":
        return train_ppo_curriculum.main()
    if args.command == "train_ppo_reward_shaped":
        return train_ppo_reward_shaped.main()
    if args.command == "train_ppo_imitation":
        return train_ppo_imitation.main()
    if args.command == "train_ppo_multiseed":
        return train_ppo_multiseed.main()
    if args.command == "train_ppo_bc_curriculum":
        return train_ppo_bc_curriculum.main()
    if args.command == "eval_checkpoints":
        return eval_checkpoints.main()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(run())
