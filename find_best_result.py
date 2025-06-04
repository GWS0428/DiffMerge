import os
import re

def parse_scores(filepath):
    """Parses tome_results.txt and returns a tuple of (CLIP score, CDS score)."""
    with open(filepath, 'r') as f:
        content = f.read()
    clip_match = re.search(r"Average CLIP Score:\s*([\d.]+)", content)
    cds_match = re.search(r"Average CDS Score:\s*([\d.]+)", content)

    if clip_match and cds_match:
        clip_score = float(clip_match.group(1))
        cds_score = float(cds_match.group(1))
        return clip_score, cds_score
    return None, None

def find_best_results(root_dir):
    best_clip = -float('inf')
    best_cds = -float('inf')
    best_clip_data = None
    best_cds_data = None

    for subdir, _, files in os.walk(root_dir):
        if 'tome_results.txt' in files:
            result_path = os.path.join(subdir, 'tome_results.txt')
            clip, cds = parse_scores(result_path)

            if clip is not None and cds is not None:
                if clip > best_clip:
                    best_clip = clip
                    best_clip_data = (subdir, clip, cds)

                if cds > best_cds:
                    best_cds = cds
                    best_cds_data = (subdir, clip, cds)

    print("Best CLIP Score:")
    if best_clip_data:
        print(f"  Directory: {best_clip_data[0]}")
        print(f"  CLIP Score: {best_clip_data[1]}")
        print(f"  CDS Score: {best_clip_data[2]}")

    print("\nBest CDS Score:")
    if best_cds_data:
        print(f"  Directory: {best_cds_data[0]}")
        print(f"  CLIP Score: {best_cds_data[1]}")
        print(f"  CDS Score: {best_cds_data[2]}")

if __name__ == "__main__":
    root_directory = "./demo"  # Replace with your actual root directory
    find_best_results(root_directory)
