"""CLI for offline trajectory HTML export."""

import argparse

from .report import export_trajectory_html


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export trajectory JSON/JSONL or a rollout directory to offline HTML")
    parser.add_argument("source", help="Trajectory JSON, pipeline JSONL, JSON array, or rollout directory")
    parser.add_argument("-o", "--output", required=True, help="Standalone output HTML")
    parser.add_argument("--task", dest="task_file", help="Optional Task JSON (public fields only)")
    parser.add_argument("--summary", dest="summary_file", help="Optional run summary JSON")
    parser.add_argument("--tasks", dest="tasks_dir", help="Optional JsonTaskStore directory")
    parser.add_argument("--title", default="Trajectory Explorer")
    parser.add_argument("--trajectory-key", default="trajectory")
    parser.add_argument("--score-key", default="traj_overall")
    parser.add_argument("--score-prefix", default="traj_")
    parser.add_argument("--force", action="store_true", help="Replace an existing HTML report, never an input")
    args = vars(parser.parse_args(argv))
    try:
        path = export_trajectory_html(**args)
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"Export failed: {exc}\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
