This repository uses Slurm batch scripts to launch bootstrap experiments.
## Step1: Launch a bootstrap script with ` sbatch`.
Use the following commands from the terminal:
```bash
sbatch run_bootstrap_shapley.sh
sbatch run_bootstrap_es.sh
sbatch run_bootstrap_lsp.sh
sbatch run_bootstrap_cons.sh
sbatch run_bootstrap_sol.sh
```
These scripts launch the bootstrap runs. In the standard setup, the bootstrap baseline uses B = 1,000 independent bootstrap replications.

## Step2: Retrieve the latest matching Slurm job ID.

To retrieve the most recent job ID associated with a specific launcher script, use:

```bash
pattern='run_bootstrap_es.sh'
start='2026-02-01'
jid=$(
  (sacct --helpformat 2>/dev/null | grep -qw SubmitLine && \
    sacct -u "$USER" -S "$start" -n -X -o JobIDRaw,SubmitLine%300 | \
    grep -F "$pattern" | awk '{print $1}' | sort -n | tail -n 1) || \
  (sacct -u "$USER" -S "$start" -n -X -o JobIDRaw,JobName%300 | \
    grep -F "$pattern" | awk '{print $1}' | sort -n | tail -n 1)
)
echo "Latest job ID: $jid"
```
The  job ID returned by `sacct` is not always the parent array job ID. For array jobs, inspect the record with  and extract the parent ID before computing the full runtime using the following:

```bash
sacct -j 6703812 --format=JobID%30,JobName%20,Start,End --noheader
```
Example output:
```bash
6700619_998    bootstrap_es_test 2026-03-19T20:20:04 2026-03-19T20:22:38
6700619_998.batch                batch 2026-03-19T20:20:04 2026-03-19T20:22:38
6700619_998.extern               extern 2026-03-19T20:20:04 2026-03-19T20:22:38
6700619_998.0                    python 2026-03-19T20:20:13 2026-03-19T20:22:38
```



## Step3: Use the job ID to inspect timing.

Once you have a job ID (example: 6700619), you can compute the wall-time span of the array with:

```bash
jid=6700619

sacct -j "$jid" --format=JobID,Start,End --noheader | \
awk -v jid="$jid" '
  $1 ~ ("^"jid"_[0-9]+$") {
    if (min == "" || $2 < min) min = $2
    if (max == "" || $3 > max) max = $3
  }
  END {
    gsub("T", " ", min)
    gsub("T", " ", max)
    "date -d \"" min "\" +%s" | getline s
    close("date -d \"" min "\" +%s")
    "date -d \"" max "\" +%s" | getline e
    close("date -d \"" max "\" +%s")
    d = e - s
    printf("Array wall time: %02d:%02d:%02d (Earliest=%s Latest=%s)\n", int(d/3600), int((d%3600)/60), d%60, min, max)
  }'
```

If your goal is to measure how long the job waited in the queue before starting, use Submit and Start instead:

```bash
jid=6700619

sacct -j "$jid" --format=JobID,Submit,Start --noheader | \
awk -v jid="$jid" '
  $1 == jid {
    subm = $2
    st = $3
  }
  END {
    gsub("T", " ", subm)
    gsub("T", " ", st)
    "date -d \"" subm "\" +%s" | getline s
    close("date -d \"" subm "\" +%s")
    "date -d \"" st "\" +%s" | getline e
    close("date -d \"" st "\" +%s")
    d = e - s
    printf("Queue wait time: %02d:%02d:%02d (Submit=%s Start=%s)\n", int(d/3600), int((d%3600)/60), d%60, subm, st)
  }'
```

