#! /bin/bash

##################################################################################################################################
# Change to suit your needs

#SBATCH --partition=shared
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
###################################################################################################################################

# Launch model server, send back server URL and wait so that SLURM does not cancel the allocation.

function get_available_port {
    # Define the range of ports to select from
    MIN_PORT=1024
    MAX_PORT=65535

    # Generate a random port number
    port=$(shuf -i $MIN_PORT-$MAX_PORT -n 1)

    # Check if the port is in use
    until ./is_port_free $port; do
        # If the port is in use, generate a new random port number
        port=$(shuf -i $MIN_PORT-$MAX_PORT -n 1)
    done

    echo $port
}

port=$(get_available_port)
export PORT=$port

####################################################################################################################################
# Only change commands within this section

# Load modules
module purge
module load gcc/13.2 openmpi/4.1.5 hdf5 netcdf cmake/3.30.5

. ~/.bashrc
conda activate python3.12

# Python script that gets node usage (Optional)
# python ~/nobackup/ExaHyPE2_UQ/umbridge/hpc/cpu_ram_log.py -u mghw54 -o ${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.log --interval=5 &

# Launch UM-Bridge server
python -u ~/nobackup/ExaHyPE2_UQ/FWI_tinyda_server.py &

#####################################################################################################################################
host=$(hostname -I | awk '{print $1}')

echo "Waiting for model server to respond at $host:$port..."
while ! curl -s "http://$host:$port/Info" > /dev/null; do
    sleep 1
done
echo "Model server responded"

# Write server URL to file identified by HQ job ID.
mkdir -p $UMBRIDGE_LOADBALANCER_COMM_FILEDIR
echo "http://$host:$port" > "$UMBRIDGE_LOADBALANCER_COMM_FILEDIR/url-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.txt"

sleep infinity # keep the job occupied
