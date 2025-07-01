# available editing methods combination: 
    # [ddim, null-text-inversion, negative-prompt-inversion, directinversion] + 
    # [p2p, masactrl, pix2pix_zero, pnp]

# p2p only!
conda run -n sdxl_null --no-capture-output python -u run_with_bench.py \
                --output_path ./output \
                --edit_category_list 0 1 2 3 4 5 6 7 8 9\
                --edit_method_list ddim+p2p negative-prompt-inversion+p2p directinversion+p2p null-text-inversion+p2p \
                --pick_image 0 4 5 6

# # pnp only!
conda run -n pnp --no-capture-output python -u run_with_bench.py \
                --output_path ./output \
                --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
                --edit_method_list ddim+pnp negative-prompt-inversion+pnp directinversion+pnp null-text-inversion+pnp \
                --pick_image 3 7 8 9

# masactrl only!
conda run -n masactrl --no-capture-output python -u run_with_bench.py \
                --output_path ./output \
                --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
                --edit_method_list ddim+masactrl negative-prompt-inversion+masactrl directinversion+masactrl null-text-inversion+masactrl \
                --pick_image 0 1 2 3