source ~/setup_pisa.sh
cd retro/retro/
./simple_likelihood.py --index $1 --file test_event$2.pkl --outdir /gpfs/scratch/pde3/retro/test_event$2/out/
