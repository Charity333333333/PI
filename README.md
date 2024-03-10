# PI
## Requirement
```
pip install fairseq==0.10.2
conda install scikit-learn
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

## Dataset
[[English-German](https://drive.google.com/file/d/1C2T9PfJnvYXT8ro-fjI5VSqklCpfzCve/view?usp=drive_link)] 

[[Chinese-English](https://drive.google.com/file/d/1x-_9GO-OuGctjY2_5B4CG6qyrJWBoBh-/view?usp=drive_link)]


## Model Checkpoints
[[English-German](https://mega.nz/file/RWNCFTZK#SO_B2mE6niOlQFvB2YFujROnGbeel9rHNmxdmQz-6_0)] 

[[Chinese-English](https://mega.nz/file/8H11hCKD#YNxX47j8H2C3mNX9_1uXM9Rm-T2ZlRV6JOWlS4yj24c)] 

## Usage

1.Download Dataset (already processed using fairseq), Model Checkpoints.

2.Training:
```
# From English to German
fairseq-train data-bin/en --user-dir . --max-tokens 4096   \
              --task parameter_inheritance_task --lang-pairs en-de,en-em,en-iw,en-no  \
              --arch parameter_inheritance_model  \
              --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9,0.98)' --clip-norm 1.0  \
              --lr-scheduler inverse_sqrt --stop-min-lr 1e-9 --warmup-init-lr 1e-7 --warmup-updates 8000 \
              --lr 5e-4 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.1 \
              --attention-dropout 0.3 --weight-decay 0.0 --max-tokens 4096 --update-freq 8 --fp16 \
              --save-dir checkpoints/en --tensorboard-logdir checkpoints/en  --log-format

# From Chinese to English 
fairseq-train data-bin/zh--user-dir . --max-tokens 4096   \
              --task parameter_inheritance_task --lang-pairs zh-en,zh-ed,zh-sp,zh-th  \
              --arch parameter_inheritance_model  \
              --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9,0.98)' --clip-norm 1.0  \
              --lr-scheduler inverse_sqrt --stop-min-lr 1e-9 --warmup-init-lr 1e-7 --warmup-updates 8000 \
              --lr 5e-4 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.1 \
              --attention-dropout 0.3 --weight-decay 0.0 --max-tokens 4096 --update-freq 8 --fp16 \
              --save-dir checkpoints/zh --tensorboard-logdir checkpoints/zh  --log-format
```

3.Decoding：
```
# From English to German
fairseq-generate $data_path --user-dir . --task parameter_inheritance_task \
               --lang-pairs en-de,en-em,en-no,en-iw  --source-lang en --target-lang de \
               --max-tokens 4096 --beam 4 --lenpen 0.6  --quiet \
               --path $model_path > generate/en2de.txt

fairseq-generate $data_path --user-dir . --task parameter_inheritance_task \
               --lang-pairs en-de,en-em,en-no,en-iw   --source-lang en --target-lang em \
               --max-tokens 4096 --beam 4 --lenpen 0.6  --quiet \
               --path $model_path > generate/en2em.txt

fairseq-generate $data_path --user-dir . --task parameter_inheritance_task \
               --lang-pairs en-de,en-em,en-no,en-iw   --source-lang en --target-lang no \
               --max-tokens 4096 --beam 4 --lenpen 0.6  --quiet \
               --path $model_path > generate/en2no.txt
               
fairseq-generate $data_path --user-dir . --task parameter_inheritance_task \
               --lang-pairs en-de,en-em,en-no,en-iw   --source-lang en --target-lang iw \
               --max-tokens 4096 --beam 4 --lenpen 0.6  --quiet \
               --path $model_path > generate/en2iw.txt

# From Chinese to English
fairseq-generate $data_path --user-dir . --task parameter_inheritance_task \
               --lang-pairs zh-en,zh-ed,zh-sp,zh-th  --source-lang zh --target-lang en \
               --max-tokens 4096 --quiet \
               --path $model_path > generate/zh2en.txt
         
fairseq-generate $data_path --user-dir . --task parameter_inheritance_task \
               --lang-pairs zh-en,zh-ed,zh-sp,zh-th  --source-lang zh --target-lang ed \
               --max-tokens 4096 --quiet \
               --path $model_path > generate/zh2ed.txt
               
fairseq-generate $data_path --user-dir . --task parameter_inheritance_task \
               --lang-pairs zh-en,zh-ed,zh-sp,zh-th  --source-lang zh --target-lang sp \
               --max-tokens 4096 --quiet \
               --path $model_path > generate/zh2sp.txt
               
fairseq-generate $data_path --user-dir . --task parameter_inheritance_task \
               --lang-pairs zh-en,zh-ed,zh-sp,zh-th  --source-lang zh --target-lang th \
               --max-tokens 4096 --quiet \
               --path $model_path > generate/zh2th.txt
```
