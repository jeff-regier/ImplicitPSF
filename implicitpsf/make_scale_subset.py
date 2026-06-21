"""Build a training-set-size subset of an existing manifest, holding val/test identical.

Only the train split is downsampled (deterministically by sorted exposure id) to N exposures;
val, test, reserved, and all other fields are copied verbatim. This isolates training-set SIZE
as the only variable for the scale curve (same sim, same PSF, same architecture, same tuning).
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    exposures = manifest["exposures"]
    train_ids = sorted(eid for eid, info in exposures.items() if info.get("split") == "train")
    if args.n_train > len(train_ids):
        raise ValueError(f"requested {args.n_train} train but only {len(train_ids)} available")
    keep_train = set(train_ids[: args.n_train])
    kept = {
        eid: info
        for eid, info in exposures.items()
        if info.get("split") != "train" or eid in keep_train
    }
    out_manifest = {**manifest, "exposures": kept}
    with open(args.out, "w") as fh:
        json.dump(out_manifest, fh, indent=2)
    n_val = sum(1 for i in kept.values() if i.get("split") == "val")
    n_test = sum(1 for i in kept.values() if i.get("split") == "test")
    print(f"wrote {args.out}: train={args.n_train} val={n_val} test={n_test}")


if __name__ == "__main__":
    main()
