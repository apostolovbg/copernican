# DEV NOTE (v1.5i hotfix 11): Accept optional covariance file argument for
# Pantheon+ dataset and forward it to run_analysis_once.
# Helper script invoked by the Flask UI. It runs copernican.run_analysis_once
# with command-line arguments so the analysis can be executed in a subprocess.
import sys
from copernican import run_analysis_once

if __name__ == "__main__":
    if len(sys.argv) not in (8, 9):
        raise SystemExit(
            "Usage: web_runner.py model_json engine sne_file bao_file "
            "sne_format bao_format output_dir [sne_cov_file]"
        )
    cov_file = sys.argv[8] if len(sys.argv) == 9 else None
    run_analysis_once(
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
        sys.argv[5], sys.argv[6], output_dir=sys.argv[7],
        sne_cov_file=cov_file,
    )

