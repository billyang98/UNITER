USERNAME = $1

cd /scratch/cluster/$USERNAME

mkdir -p bigtmpdir
mkdir -p singularity_cache

TMPDIR=/scratch/cluster/$USERNAME/bigtmpdir SINGULARITY_CACHEDIR=/scratch/cluster/$USERNAME/singularity_cache singularity build -s uniter_image docker://chenrocks/uniter

