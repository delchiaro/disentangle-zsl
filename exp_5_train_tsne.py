
from exp_5 import exp, init_exp
from exp_5_tsne import tsne


def main():
    exp_name = 'hello'
    model, train, val, test_unseen, test_seen = init_exp(0, 42, use_valid=False)
    epochs = 30

    model, _, _, test_accs = exp(model, train, val, test_unseen, test_seen,
                                 nb_epochs=epochs,
                                 use_masking=True,
                                 attrs_key='class_attr', mask_attrs_key='class_attr',
                                 a_lr=.000002, c_lr=.0001,
                                 pretrain_cls_epochs=0,
                                 infinite_dataset=True,  # False

                                 test_lr=.0004, test_gen_samples=2000,  # 800
                                 test_period=1, test_epochs=10,
                                 test_encode=False, test_use_masking=True,

                                 verbose=3,
                                 state_dir=f'states/{exp_name}')

    for i in range(0, epochs+1, 1):
        s = model.load(epoch=i)
        app_title = f'- ep={i} - acc={s["adapt_acc"]*100:2.2f}'
        tsne(model, [test_unseen, train], train, infinite_dataset=True, append_title=app_title, savepath=f'tsne/{exp_name}', target='feats')
        tsne(model, [test_unseen, train], train, infinite_dataset=True, append_title=app_title, savepath=f'tsne/{exp_name}', target='attr_emb')



if __name__ == '__main__':
    main()