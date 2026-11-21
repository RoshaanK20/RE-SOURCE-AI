"""Create a DRAFT scientific contract. Researcher must verify every field before approval.

Example:
python -m scripts.make_science_contract --source ./research_repo --evaluator ./eval.py \
 --paper-title 'Research paper title' --repo-url https://github.com/owner/repo \
 --commit 0123456789abcdef0123456789abcdef01234567 \
 --protect config/train.json --protect data/labels.csv --out ./contracts/paper.json
"""
import argparse
import json
from pathlib import Path

from resource_ai.science import sha, reference_digest, _relpath


def main(argv=None):
    p = argparse.ArgumentParser(description='Generate a draft, framework-independent paper methodology contract')
    p.add_argument('--source', required=True)
    p.add_argument('--evaluator', required=True)
    p.add_argument('--paper-title', required=True)
    p.add_argument('--repo-url', required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--framework', default='unspecified')
    p.add_argument('--protect', action='append', required=True,
                   help='Dataset, split, preprocessing, model configuration or evaluation file')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    original = Path(args.source).resolve(strict=True)
    evaluator = Path(args.evaluator).resolve(strict=True)
    if evaluator.is_relative_to(original):
        p.error('Evaluator must live outside the original source tree')
    protected = {}
    for raw in args.protect:
        rel = _relpath(raw)
        candidate = (original / rel).resolve(strict=True)
        if not candidate.is_relative_to(original) or not candidate.is_file():
            p.error('Protected path must be a regular file inside original project: ' + rel)
        protected[rel] = sha(candidate)
    if not protected:
        p.error('At least one protected dataset/config file is required')
    contract = {
        'schema': 'resource.scientific-contract.v1',
        'review_status': 'draft',
        'paper': {'title': args.paper_title, 'repository_url': args.repo_url, 'commit': args.commit},
        'framework': args.framework,
        'methodology': {k:{'description':'REVIEW_ME'} for k in
                        ('dataset','split','preprocessing','model','training','evaluation')},
        'reference_tree_sha256': reference_digest(original),
        'trusted_evaluator_sha256': sha(evaluator),
        'protected_files': protected,
        'evidence_files': {k:[] for k in
                           ('dataset','split','preprocessing','model','training','evaluation')},
        'approved_changes': [],
        'expected_metric': {'name':'REVIEW_ME', 'value':0.0, 'abs_tolerance':0.0},
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    print('DRAFT contract written:', dest)
    print('Review the methodology against the original paper; fill metric/tolerance and approved changes;')
    print('set review_status="approved" only after independent researcher review.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
