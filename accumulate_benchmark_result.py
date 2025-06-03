import os
import argparse

def parse_metrics(file_path):
    """Parses a metric file and returns the CLIP and CDS scores as floats."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    clip_score = float([line for line in lines if "CLIP Score" in line][0].split(":")[1].strip())
    cds_score = float([line for line in lines if "CDS Score" in line][0].split(":")[1].strip())
    return clip_score, cds_score

def accumulate_scores(dataset_name, version):
    """Accumulates CLIP and CDS scores from metric files."""
    total_clip = 0.0
    total_cds = 0.0
    count = 0

    # current file path 
    cur_file_dir_path = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(cur_file_dir_path, 'demo', dataset_name)
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(f"{version}_metrics.txt"):
                file_path = os.path.join(root, file)
                clip, cds = parse_metrics(file_path)
                total_clip += clip
                total_cds += cds
                count += 1

    if count == 0:
        raise ValueError(f"No {version}_metric.txt files found in {dataset_dir}")

    avg_clip = total_clip / count
    avg_cds = total_cds / count
    return avg_clip, avg_cds

def write_results(dataset_name, version, avg_clip, avg_cds):
    """Writes the average scores to a result file."""
    cur_file_dir_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(cur_file_dir_path, 'demo', dataset_name, f"{version}_results.txt")
    with open(result_path, "w") as f:
        f.write(f"Average CLIP Score: {avg_clip:.4f}\n")
        f.write(f"Average CDS Score: {avg_cds:.8f}\n")
    print(f"Results written to {result_path}")

def main():
    parser = argparse.ArgumentParser(description="Summarize diffusion model benchmark results.")
    parser.add_argument("--dataset_name", type=str, help="Path to the dataset directory (e.g., ./wild-ti2i)")
    parser.add_argument("--version", choices=["tome", "standard"], help="Version to evaluate (tome or standard)")
    args = parser.parse_args()

    avg_clip, avg_cds = accumulate_scores(args.dataset_name, args.version)
    write_results(args.dataset_name, args.version, avg_clip, avg_cds)

if __name__ == "__main__":
    main()
