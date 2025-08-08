# rm -rf /data/zhangweiqi/my_Text2Tex/outputs/try/44-p40-h20-1.0-0.6-0.1/update/mask
export GPU=4

export CUDA_VISIBLE_DEVICES=$GPU
python scripts/generate_gaussian_text.py \
    --input_dir  data/backpack\
    --output_dir outputs/backpack\
    --pc_name backpack \
    --pc_file backpack.ply \
    --prompt "An orange backpack." \
    --ddim_steps 50 \
    --new_strength 1 \
    --update_strength 0.6 \
    --blend 0 \
    --dist 1 \
    --num_viewpoints 40 \
    --viewpoint_mode predefined \
    --use_principle \
    --update_steps 30 \
    --update_mode heuristic \
    --seed 43 \
    --device "2080" \
    --use_objaverse

# python ol.py 5 6