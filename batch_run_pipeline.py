#!/usr/bin/env python3
"""
Batch Run Pipeline Script

This script automatically scans a directory of folders, matches cam0 and cam1 videos
inside each folder based on their unique identifier (timestamp) code, and executes the
CV object identification pipeline sequentially for each pair.

Features:
- Processes one folder at a time.
- Matches video pairs using a unique identifier code at the end of filenames.
- Generates session IDs in the format: <unique_code>_<folder_name>.
- Forwards any additional arguments directly to main.py (e.g., --device, --det_conf).
- Logs overall success/failure statistics.
"""

import os
import sys
import re
import argparse
import subprocess
from pathlib import Path

# Regex to match cam0 output filename and capture the unique identifier at the end.
# Matches format: cam0_output_<unique_code>.<ext> or cam0_<unique_code>.<ext>
CAM0_REGEX = re.compile(r'cam0(?:_output)?_(.+)\.(mp4|avi|mkv|mov|flv|webm)$', re.IGNORECASE)

def sanitize_name(name: str) -> str:
    """Sanitizes folder name to be safe for filenames and session IDs."""
    # Replace spaces and other non-alphanumeric/dash chars with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_')

def find_video_pairs(folder_path: Path):
    """
    Finds and pairs cam0 and cam1 video files in a folder.
    
    Returns:
        A list of dictionaries containing:
        - 'unique_code': The extracted identifier
        - 'cam0_path': Absolute path to cam0 video
        - 'cam1_path': Absolute path to cam1 video
    """
    pairs = []
    if not folder_path.is_dir():
        return pairs

    files = sorted(os.listdir(folder_path))
    for f in files:
        match = CAM0_REGEX.match(f)
        if match:
            unique_code = match.group(1)
            ext = match.group(2)
            
            # Formulate the corresponding cam1 filename
            cam1_filename = f.replace("cam0", "cam1")
            cam1_path = folder_path / cam1_filename
            
            # Check if cam1 file exists in the folder
            if cam1_filename in files:
                pairs.append({
                    'unique_code': unique_code,
                    'cam0_path': str(folder_path / f),
                    'cam1_path': str(cam1_path)
                })
            else:
                # Fallback check (case-insensitive)
                found = False
                for other_file in files:
                    if other_file.lower() == cam1_filename.lower():
                        pairs.append({
                            'unique_code': unique_code,
                            'cam0_path': str(folder_path / f),
                            'cam1_path': str(folder_path / other_file)
                        })
                        found = True
                        break
                if not found:
                    print(f"Warning: Found cam0 video '{f}' but no matching cam1 video in '{folder_path}'")
                    
    return pairs

def main():
    parser = argparse.ArgumentParser(
        description="Batch run the CV Object Identification Pipeline on matched video pairs across directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-d", "--dir",
        type=str,
        required=True,
        help="Root directory containing subdirectories of videos to process (e.g., /home/vickicv/CV-Testing/priyanshue5-long)"
    )
    
    parser.add_argument(
        "-p", "--pipeline_dir",
        type=str,
        default="/home/vickicv/CV-Testing/testing/viatouch-cv-object-identification-pipeline",
        help="Path to the directory containing main.py"
    )
    
    parser.add_argument(
        "-e", "--executable",
        type=str,
        default=sys.executable,
        help="Python executable to run main.py"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run: scan directories and print video pairs without executing the pipeline"
    )

    parser.add_argument(
        "--swap_cameras",
        action="store_true",
        help="Pass cam1 to --video0 and cam0 to --video1"
    )

    # Parse args, allowing any unknown args to be forwarded directly to main.py
    args, extra_args = parser.parse_known_args()
    
    root_dir = Path(args.dir).resolve()
    pipeline_dir = Path(args.pipeline_dir).resolve()
    main_py_path = pipeline_dir / "main.py"
    
    # Validate directories
    if not root_dir.exists():
        print(f"Error: Root directory '{root_dir}' does not exist.")
        sys.exit(1)
        
    if not args.dry_run and not main_py_path.exists():
        print(f"Error: main.py not found at '{main_py_path}'. Please specify the correct --pipeline_dir.")
        sys.exit(1)
        
    print("=" * 60)
    print(f"BATCH RUN PIPELINE CONFIGURATION")
    print(f"Root Directory: {root_dir}")
    print(f"Pipeline Dir:   {pipeline_dir}")
    print(f"Executable:     {args.executable}")
    if extra_args:
        print(f"Forwarded Args: {' '.join(extra_args)}")
    if args.dry_run:
        print("NOTE: RUNNING IN DRY-RUN MODE")
    print("=" * 60)

    # Check if the root directory itself directly contains video pairs
    pairs_in_root = find_video_pairs(root_dir)
    
    if pairs_in_root:
        # If the root directory directly contains matched videos, process it as the single target folder
        dirs_to_process = [root_dir]
        print(f"Direct video pairs found in the target directory itself. Processing target directory directly.")
    else:
        # Otherwise, scan subdirectories
        dirs_to_process = sorted([d for d in root_dir.iterdir() if d.is_dir()])
        if not dirs_to_process:
            print("No video pairs or subdirectories found in the specified directory.")
            sys.exit(0)
        print(f"No direct video pairs found in the target directory. Scanning {len(dirs_to_process)} subdirectories.")
        
    runs_total = 0
    runs_success = 0
    runs_failed = []
    
    # Process one folder at a time
    for folder_path in dirs_to_process:
        folder_name = folder_path.name
        print(f"\nScanning folder: {folder_name}")
        
        # Avoid running find_video_pairs again if we already scanned root
        if folder_path == root_dir:
            pairs = pairs_in_root
        else:
            pairs = find_video_pairs(folder_path)
        if not pairs:
            print(f"  No valid video pairs found in '{folder_name}'.")
            continue
            
        print(f"  Found {len(pairs)} video pair(s) in '{folder_name}':")
        for pair in pairs:
            print(f"    - [{pair['unique_code']}] {Path(pair['cam0_path']).name} <-> {Path(pair['cam1_path']).name}")
            
        for pair in pairs:
            unique_code = pair['unique_code']
            sanitized_folder = sanitize_name(folder_name)
            session_id = f"{unique_code}_{sanitized_folder}"
            
            print(f"\n--> Starting processing for: folder='{folder_name}', unique_code='{unique_code}'")
            print(f"    Generated Session ID: {session_id}")
            
            # Construct execution command
            if args.swap_cameras:
                cmd = [
                    args.executable,
                    str(main_py_path),
                    "--video0", pair['cam1_path'],
                    "--video1", pair['cam0_path'],
                    "--session_id", session_id
                ]
            else:
                cmd = [
                    args.executable,
                    str(main_py_path),
                    "--video0", pair['cam0_path'],
                    "--video1", pair['cam1_path'],
                    "--session_id", session_id
                ]
            # Append forwarded arguments
            cmd.extend(extra_args)
            
            runs_total += 1
            
            if args.dry_run:
                print(f"    [DRY-RUN] Command: {' '.join(cmd)}")
                runs_success += 1
            else:
                print(f"    Executing command: {' '.join(cmd)}")
                try:
                    # Execute subprocess in pipeline_dir
                    result = subprocess.run(cmd, cwd=pipeline_dir, check=False)
                    if result.returncode == 0:
                        print(f"    Success!")
                        runs_success += 1
                    else:
                        print(f"    Failed! (Exit code: {result.returncode})")
                        runs_failed.append({
                            'folder': folder_name,
                            'unique_code': unique_code,
                            'session_id': session_id,
                            'exit_code': result.returncode
                        })
                except Exception as e:
                    print(f"    Execution encountered exception: {e}")
                    runs_failed.append({
                        'folder': folder_name,
                        'unique_code': unique_code,
                        'session_id': session_id,
                        'error': str(e)
                    })

    # Summary report
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETED SUMMARY")
    print(f"Total Runs attempted:  {runs_total}")
    print(f"Successful Runs:       {runs_success}")
    print(f"Failed Runs:           {len(runs_failed)}")
    
    if runs_failed:
        print("\nFailed Executions List:")
        for fail in runs_failed:
            err = fail.get('exit_code')
            if err is not None:
                err_str = f"Exit code {err}"
            else:
                err_str = f"Error: {fail.get('error')}"
            print(f"  - Folder: '{fail['folder']}', Code: {fail['unique_code']}, Session ID: {fail['session_id']} ({err_str})")
    print("=" * 60)

if __name__ == "__main__":
    main()
