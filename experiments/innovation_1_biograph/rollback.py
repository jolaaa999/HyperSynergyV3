# [Innovation-1: Biological Graph Cell Encoder]
"""回到本任务开始前的工作区内容，保留用户原有未提交修改及全部实验记录。"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='不加参数时只核验并列出，不回滚')
    args = parser.parse_args()
    audit = json.loads((EXP/'results/implementation_audit.json').read_text())
    expected = json.loads((EXP/'results/modified_source_sha256.json').read_text())
    for name, value in expected.items():
        path = ROOT/name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != value:
            raise RuntimeError('Source edited since implementation; refusing to overwrite: ' + name)
    print('Restore originals:', audit['modified_existing_sources'])
    print('Remove only new core sources:', audit['added_core_sources'])
    print('Keep all experiments, original pre-existing edits, checkpoints and logs.')
    if args.apply:
        for name in audit['modified_existing_sources']:
            shutil.copy2(str(EXP/'originals'/name), str(ROOT/name))
        for name in audit['added_core_sources']:
            (ROOT/name).unlink()
        print('Restored pre-task working tree sources.')
    else:
        print('DRY RUN; no files changed.')


if __name__ == '__main__':
    main()
