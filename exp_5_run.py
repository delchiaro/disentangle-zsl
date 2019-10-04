from typing import Union

from exp_5 import exp, init_exp, Model, AutoencoderHiddens, LossMult


def build_model(feats_dim: int, nb_attributes: int, nb_train_classes: int, nb_test_classes: int, device: Union[str, None] = None) -> Model:
    auto_hiddens = AutoencoderHiddens()
    cls_hiddens = (2048,)
    cntx_cls_hiddens = None
    return Model(feats_dim, nb_attributes, nb_train_classes, nb_test_classes, z_dim=2048, ka_dim=32, kc_dim=256,  # best kc: 256, 512, 768
                 autoencoder_hiddens=auto_hiddens, cls_hiddens=cls_hiddens, cntx_cls_hiddens=cntx_cls_hiddens,

                 conv1by1=True,
                 use_context=True,
                 device=device)
def main():
    model, train, test_unseen, test_seen, val = init_exp(build_model, gpu=0, seed=None)
    loss_mult = LossMult(rec_x=5,
                         rec_a=5,
                         rec_ac=1,

                         cls_x=.01,
                         cls_x1=.01,
                         cls_cntx=.001,
                         rec_ad=1)


    _, _, test_accs = exp(model, train, test_unseen, loss_mult=loss_mult, bs=128, nb_epochs=40,
                          a_lr=.000002,
                          c_lr=.0001, pretrain_cls_epochs=0,


                          train_frankenstain=True,
                          bin_attrs=False,
                          bin_masks=False,  # True
                          masking=True,


                          test_period=1, test_epochs=10, test_lr=.0001, test_gen_samples=400,


                          attr_disentangle=False,

                          early_stop_patience=None,
                          test_tsne=False,
                          verbose=3,
                          state_dir='states/exp_5_1')


if __name__ == '__main__':
    main()