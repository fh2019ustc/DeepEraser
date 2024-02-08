import gradio as gr
import torch
import numpy as np
from PIL import Image
from model import DeepEraser
import glob
from gradio.components import Image as grImage

def reload_rec_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


rec_model_path = './deeperaser.pth'
net = DeepEraser().cuda()
reload_rec_model(net, rec_model_path)
net.eval()


def image_mod(images):
    try:
        img = np.array(images["image"])[:, :, :3]
        mask = np.array(images["mask"].convert('L'))[:, :]

        im = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask / 255.0).unsqueeze(0).float()

        im_mask = torch.cat([im, mask], dim=0).unsqueeze(0)

        with torch.no_grad():
            pred_img = net(im.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda())
            pred_img[-1] = torch.clamp(pred_img[-1], 0, 1)

            out = (pred_img[-1][0]*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        return Image.fromarray(out)

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print('Out of memory error caught, releasing unused memory')
            torch.cuda.empty_cache()
        else:
            raise e
    finally:
        del im, mask, im_mask, pred_img
        torch.cuda.empty_cache()


demo_img_files = glob.glob('./demo_imgs/*.[jJ][pP][gG]') + glob.glob('./demo_imgs/*.[pP][nN][gG]')

demo = gr.Interface(
    image_mod,
    # gr.inputs.Image(type="pil", tool='sketch', label="Image & Mask"),
    # gr.outputs.Image(type="pil"),
    grImage(type="pil", tool='sketch', label="Image & Mask"),
    grImage(type="pil"),
    title="DeepEraser", examples=demo_img_files, allow_flagging="never"
)


if __name__ == "__main__":
    demo.launch(share=False, server_port=6212)
 

