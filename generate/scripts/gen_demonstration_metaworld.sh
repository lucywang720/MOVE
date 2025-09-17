# bash scripts/gen_demonstration_metaworld.sh basketball



cd third_party/Metaworld

task_name=${1}

export CUDA_VISIBLE_DEVICES=2


python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 10000 \
            --root_dir "data/10000" \
            --velocity_move 0.03 \
            --velocity_rotate 1 \
            --velocity_obj 0 \
            --velocity_camera 0.03 \
            --move_step 30

