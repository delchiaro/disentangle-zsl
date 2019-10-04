
from exp_5 import exp, init_exp, build_model_A, build_model_B, build_model_C
from exp_5_tsne import tsne
import shutil
import os

def main():
    gpu=0
    seed=42
    exp_name = 'modelA_frank_train_contrastive'
    build_model_fn = build_model_A
    state_dir = f'states/{exp_name}'
    tsne_dir = f'tsne/{exp_name}'

    try:
        os.makedirs(state_dir, exist_ok=False)
    except FileExistsError as e:
        print('\n\nSTATE FOLDER WITH THIS EXP_NAME ALREADY EXISTS!!')
        #raise e

    model, train, test_unseen, test_seen, val = init_exp(build_model_fn, gpu, seed, use_valid=False)
    epochs = 60
    shutil.copy('exp_5.py', os.path.join(state_dir, f'_exp_5__{exp_name}.py'))
    shutil.copy('exp_5_train_tsne.py', os.path.join(state_dir, f'_exp_5_train_tsne__{exp_name}.py'))
    _, _, test_accs = exp(model, train, test_unseen, test_seen, val,
                          nb_epochs=epochs,
                          masking=True,
                          #attrs_key='class_attr', mask_attrs_key='class_attr',
                          attrs_key='class_attr_bin', mask_attrs_key='class_attr_bin',
                          #a_lr=.000002, c_lr=.0001,
                          a_lr=.00002, c_lr=.0001,
                          pretrain_cls_epochs=0,
                          infinite_dataset=True,  # False
                          train_frankenstain=True,
                          test_lr=.001, test_gen_samples=500,  # 2000,  # 800
                          test_period=1, test_epochs=4,
                          test_encode=False, test_masking=True,

                          verbose=3,
                          state_dir=state_dir)

    for i in range(0, epochs+1, 1):
        model.load(state_dir, epoch=i)
        app_title = f'- ep={i} - acc={model.acc*100:2.2f}'
        tsne(model, [test_unseen, train], train, infinite_dataset=True, append_title=app_title, savepath=tsne_dir, target='feats')
        tsne(model, [test_unseen, train], train, infinite_dataset=True, append_title=app_title, savepath=tsne_dir, target='attr_emb')



if __name__ == '__main__':
    main()