from model.SimCLR import SimCLR_ViT
from model.SpatialMAE import SpatialMAE
from model.SS_MAE import SS_MAE
from model.HSIMAE import HSIMAE,HSIMAE_lp
from model.WSGraphCL import WSGraphCL
from model.TMAC import TMAC
from model.MSG import MSG


def get_model(args):
    if args.model == 'MSG':
        model = MSG(
            mode=args.mode,
            channel_number=args.channel_num,
            img_size=args.crop_size,
            patch_size=args.patch_size,
            encoder_dim=args.encoder_dim,
            encoder_depth=args.encoder_depth,
            encoder_heads=args.head,
            decoder_dim=args.decoder_dim,
            decoder_depth=args.decoder_depth,
            decoder_heads=args.head,
            mlp_ratio=args.mlp_ratio,
            mask_ratio=args.mask_ratio,
            mask_ratio_contrast=args.mask_ratio_contrast,
            group_size=args.group_size,
            num_classes=args.num_classes,
            pretrained_path=args.weights_path
        )
    elif args.model == 'SpatialMAE':
        model = SpatialMAE(
            mode=args.mode,
            channel_number=args.channel_num,
            img_size=args.crop_size,
            patch_size=args.patch_size,
            encoder_dim=args.encoder_dim,
            encoder_depth=args.encoder_depth,
            encoder_heads=args.head,
            decoder_dim=args.decoder_dim,
            decoder_depth=args.decoder_depth,
            decoder_heads=args.head,
            mlp_ratio=args.mlp_ratio,
            mask_ratio=args.mask_ratio,
            num_classes=args.num_classes,
            pretrained_path=args.weights_path
        )
    elif args.model == 'SimCLR':
        model = SimCLR_ViT(
            mode=args.mode,
            in_channels=args.channel_num,
            img_size=args.crop_size,
            patch_size=args.patch_size,
            encoder_dim=args.encoder_dim,
            encoder_depth=args.encoder_depth,
            encoder_heads=args.head,
            projector_out=args.encoder_dim,
            mlp_ratio=args.mlp_ratio,
            temperature=args.contrastive_temp,
            num_classes=args.num_classes,
            pretrained_path=args.weights_path
        )
    elif args.model == 'SS-MAE':
        model = SS_MAE(
            mode=args.mode,
            channel_number=args.channel_num,
            img_size=args.crop_size,
            patch_size=args.patch_size,
            encoder_dim=args.encoder_dim,
            encoder_depth=args.encoder_depth,
            encoder_heads=args.head,
            decoder_dim=args.decoder_dim,
            decoder_depth=args.decoder_depth,
            decoder_heads=args.head,
            mlp_ratio=args.mlp_ratio,
            mask_ratio=args.mask_ratio,
            num_classes=args.num_classes,
            pretrained_path=args.weights_path
        )
    elif args.model == 'HSIMAE':
        if args.mode == 'pretrain':
            model = HSIMAE(
                img_size=args.crop_size,
                patch_size=args.patch_size,
                embed_dim=args.encoder_dim,
                depth=args.encoder_depth,
                num_heads=args.head,
                bands=args.channel_num,
                b_patch_size=args.channel_num // 4,
                s_depth=args.encoder_depth // 2,
                mlp_ratio=args.mlp_ratio
            )
        else:
            model = HSIMAE_lp(
                mode=args.mode,
                img_size=args.crop_size,
                patch_size=args.patch_size,
                embed_dim=args.encoder_dim,
                depth=args.encoder_depth,
                num_heads=args.head,
                bands=args.channel_num,
                b_patch_size=args.channel_num // 4,
                s_depth=args.encoder_depth // 2,
                mlp_ratio=args.mlp_ratio,
                num_class=args.num_classes,
                pretrained_path=args.weights_path
            )
    elif args.model == 'WSGraphCL':
        model = WSGraphCL(
            mode=args.mode,
            in_channels=args.channel_num,
            encoder_dim=args.encoder_dim,
            encoder_depth=args.encoder_depth,
            proj_dim=args.decoder_dim,
            pretrained_path=args.weights_path,
            num_classes=args.num_classes
        )
    elif args.model == 'TMAC':
        model = TMAC(
            mode=args.mode,
            channel_number=args.channel_num,
            img_size=args.crop_size,
            patch_size=args.patch_size,
            encoder_dim=args.encoder_dim,
            encoder_depth=args.encoder_depth,
            encoder_heads=args.head,
            decoder_dim=args.decoder_dim,
            decoder_depth=args.decoder_depth,
            decoder_heads=args.head,
            mlp_ratio=args.mlp_ratio,
            mask_ratio=args.mask_ratio,
            num_classes=args.num_classes,
            proj_dim=args.encoder_dim,
            pretrained_path=args.weights_path
        )
    else:
        raise KeyError("{} model is not supported yet".format(args.model))
    return model
