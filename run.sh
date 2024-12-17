stage=1 # start from 0 if you need to start from data preparation
stop_stage=1
# PITCH_MEAN=216.38351440429688, PITCH_STD=59.85629653930664
# PITCH_MIN=65.4063949584961, PITCH_MAX=1223.1280517578125
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
        # python examples/tts/fastpitch.py
    BRANCH='r1.19.1'
fi
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # python chinese_pitch.py
    # mkdir DataChinese && \
        # cd DataChinese && \
        # ngc registry resource download-version "nvidia/sf_bilingual_speech_zh_en:v1" && \
        # cd sf_bilingual_speech_zh_en_vv1 && \
        # unzip SF_bilingual.zip
    # DataChineseTTS directory looks like
    # ls NeMoChineseTTS/DataChinese/sf_bilingual_speech_zh_en_vv1/SF_bilingual
    # python get_data.py \
    #         --data-root /Data/***/Data/Aishell/data_aishell/wav \
    #         --manifests-path ./ \
    #         --val-size 0.005 \
    #         --test-size 0.01
    # python ./data/Ljspeech/get_data.py \
    #         --data-root /Data/***/Data/Aishell/data_aishell/wav/test \
    #         --manifests-path /Data/***/Data/Aishell/data_aishell/wav \
    #         --val-size 0 \
    #         --test-size 0
    # ls NeMoChineseTTS/*.json
    # python ./data/Librispeech/get_librispeech_data.py 
    python ./data/Libritts/extract_sup_data.py  #ds_for_fastpitch_align
    python ./data/Libritts/get_data.py --data-root /Data/***/Wetts/fastspeech2/memo/data/libritts \
    #   --data-sets "ALL"

fi


# 开始训练
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

    CUDA_VISIBLE_DEVICES=0,1 python fastpitch.py --config-path ./conf --config-name fastpitch_align_v1.05 \

fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python ../scripts/dataset_processing/tts/resynthesize_dataset.py \
        --model-path /Data/***/Wetts/fastspeech2/memo/exdata/fast-pitch-En/Libritts_100/2024-06-24_20-27-29/checkpoints/Libritts_100--val_loss=1.2161-epoch=109.ckpt \
        --input-json-manifest "/Data/***/Wetts/fastspeech2/memo/data/libritts/LibriTTS/train_clean_100_manifest.json" \
        --input-sup-data-path "/Data/***/Wetts/fastspeech2/memo/data/libritts/LibriTTS/sup_data_clean_100" \
        --output-folder "/Data/***/Wetts/fastspeech2/memo/exdata/fast-pitch-En/Libritts_100/out-put/train" \
        --device "cuda:0" \
        --batch-size 1 \
        --num-workers 1
fi
# 
# /Data/***/Wetts/fastspeech2/memo/premodel/fastpitch.nemo
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python spectrogram_enhancer_t.py 
fi
# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python spectrogram_enhancer_2nd.py
# fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    CUDA_VISIBLE_DEVICES=2,3 python spectrogram_enhancer_mou.py

fi
# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#     # CUDA_VISIBLE_DEVICES=2,3 python spectrogram_enhancer_pix_block.py 
#     CUDA_VISIBLE_DEVICES=2,3 python spectrogram_enhancer_transformer_norm.py 

# fi
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python spectrogram_enhancer_syn.py  

fi
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python spectrogram_enhancer_mapping.py  

fi

# if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
#     NEMO_ROOT="/home/***/SpeechRecognition/nemo/NeMo"
#     CUDA_VISIBLE_DEVICES=2,3 python $NEMO_ROOT/examples/tts/hifigan.py \
#         train_dataset=/Data/***/Wetts/fastspeech2/memo/exdata/fast-pitch-En/1k_all_pitch/train_manifest_mel.json \
#         validation_datasets=/Data/***/Wetts/fastspeech2/memo/exdata/fast-pitch-En/1k_1_5_pitch/val/val_manifest_mel.json \
#         model.optim.lr=0.0002 \
#         +trainer.max_epochs=300 \
#         trainer.check_val_every_n_epoch=5 \
#         trainer.devices=-1 \
#         trainer.strategy='ddp' \
#         trainer.precision=16 \
#         model.generator.upsample_initial_channel=64 \
#         exp_manager.exp_dir=/Data/***/Wetts/fastspeech2/memo/exdata/fast-pitch-En/hifigan_all_300_b32 \
#         +trainer.limit_train_batches=32 
#         # +trainer.limit_val_batches=1 \
#         # trainer.strategy=null \
#         # model.train_ds.dataloader_params.batch_size=4 \
#         # model.train_ds.dataloader_params.num_workers=0 \
#         # model.validation_ds.dataloader_params.batch_size=4 \
#         # model.validation_ds.dataloader_params.num_workers=0 \
# fi
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    CUDA_VISIBLE_DEVICES=0,1 python hifigan_inference.py \
        --finetuned_fastpitch_checkpoint=/Data/***/Wetts/fastspeech2/memo/premodel/tts_en_fastpitch.nemo \
        --finetuned_hifigan_checkpoint=/Data/***/Wetts/fastspeech2/memo/premodel/tts_hifigan.nemo \
        --valid_manifest=/Data/***/Wetts/fastspeech2/memo/data/slurp/train_main_file_clean_headset_t.json \
        --output_dir=/Data/***/Wetts/fastspeech2/memo/data/slurp/slurp_head_syn/pre_fast_all_hifilibri_19
fi
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    CUDA_VISIBLE_DEVICES=0,1 python hifigan_train.py \
   
fi
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python hifigan_inference.py \
        --finetuned_fastpitch_checkpoint=/Data/***/Wetts/fastspeech2/memo/exdata/fast-pitch-En/FastPitch_All/2024-04-09_21-25-31/checkpoints/FastPitch_All.nemo \
        --finetuned_hifigan_checkpoint=/Data/***/Wetts/fastspeech2/memo/premodel/tts_hifigan.nemo \
        --valid_manifest=/Data/***/Data/LibriSpeech/train_clean_100.json \
        --output_dir=/Data/***/Wetts/fastspeech2/memo/data/libri_syn
fi
# /Data/***/Wetts/fastspeech2/memo/data/libritts/LibriTTS/speakers.txt
# /Data/***/Wetts/fastspeech2/memo/hifigan-libritts/HifiGan/test_run/checkpoints/HifiGan--val_loss_0.1516-epoch_19-last.ckpt
