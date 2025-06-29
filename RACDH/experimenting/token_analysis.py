#!/usr/bin/env python
"""
Compute the percentage of samples where the *first token in the entity span*
is literally the same token as the *first generated token*.

Definition of “same” here = the two stored hidden-state stacks
(`torch.Tensor[num_layers, hidden_dim]`) are element-wise identical.
"""

import os
import argparse
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hiddens",
        required=True,
        help="Path to hiddens_all.pt (list[dict] with 4 keys per sample)",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------
    hidden_states = torch.load(args.hiddens, map_location="cpu")
    total = len(hidden_states)
    assert total > 0, "File is empty"

    # --------------------------------------------------------------
    # Count matches
    # --------------------------------------------------------------
    n_equal = 0
    for sample in hidden_states:
        h_ent = sample["first_token_entity"]         # Tensor (L, H)
        h_gen = sample["first_token_generation"]     # Tensor (L, H)

        if torch.equal(h_ent, h_gen):
            n_equal += 1

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------
    pct = 100.0 * n_equal / total
    print(f"Samples checked : {total}")
    print(f"Identical stacks: {n_equal}")
    print(f"Percentage      : {pct:.2f}%  "
          "(first_token_entity == first_token_generation)")


if __name__ == "__main__":
    main()
