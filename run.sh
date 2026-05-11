for n in 4 8 16 32; do
    python scaling_failure.py --active-experts=$n --num-experts=$n --datasets=Beauty --keep-checkpoints
done