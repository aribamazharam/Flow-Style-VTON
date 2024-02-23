from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # Update the paths for warp_checkpoint and gen_checkpoint arguments
        self.parser.add_argument('--warp_checkpoint', type=str, default='./lip_final.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--gen_checkpoint', type=str, default='./u2netp.pth', help='load the pretrained model from the specified location')
        # Add an additional argument for the third checkpoint
        self.parser.add_argument('--additional_checkpoint', type=str, default='./u2netp.pth.1', help='load the pretrained model from the specified location')


        self.isTrain = False
