# -*- coding: utf-8 -*-

import torch
import torchvision
import streamlit as st
from PIL import Image

import nst.torch_models
import nst.torch_utils

################################################################################

def main():
    st.markdown("# Neural Style Transfer")

    with st.spinner("Loading model ..."):
        model = _load_model("vgg19")
        # st.balloons()

    # Content image
    content_img = st.file_uploader("Content Image",
                                   type=["png", "jpg", "jpeg"],
                                   key="content_image")
    if content_img is not None:
        content_img = Image.open(content_img).convert("RGB")
        st.image(content_img, caption="Content Image")

    # Style image
    style_img = st.file_uploader("Style Image",
                                 type=["png", "jpg", "jpeg"],
                                 key="style_image")
    if style_img is not None:
        style_img = Image.open(style_img).convert("RGB")
        st.image(style_img, caption="Style Image")

    # Output size & other settings
    img_width = st.sidebar.number_input(
        label="Width", min_value=64, max_value=8192,
        value=1024 if content_img is None else content_img.width,
        key="img_width")
    img_height = st.sidebar.number_input(
        label="Height", min_value=64, max_value=8192,
        value=1024 if content_img is None else content_img.height,
        key="img_height")

    n_iterations = st.sidebar.number_input(label="# Iterations", min_value=16,
                                           max_value=1024, value=256)
    content_weight = st.sidebar.number_input(label="Content loss weight",
                                             min_value=1, max_value=1000000,
                                             value=1)
    style_weight = st.sidebar.number_input(label="Style loss weight",
                                           min_value=1, max_value=1000000,
                                           value=1000000)
    if torch.cuda.is_available():
        use_gpu = st.sidebar.checkbox("Use GPU", value=True)

    # Neural Style Transfer
    run_clicked = st.button("Run")
    if run_clicked:
        if content_img is None:
            st.text("Please upload a content image")
        if style_img is None:
            st.text("Please upload a style image")
        if content_img is not None and style_img is not None:
            _run_nst(model=model,
                     content_img=content_img.resize((img_width, img_height)),
                     style_img=style_img.resize((img_width, img_height)),
                     n_iterations=n_iterations,
                     content_weight=content_weight,
                     style_weight=style_weight,
                     use_gpu=use_gpu)


@st.cache(allow_output_mutation=True)
def _load_model(model_name) -> nst.torch_utils.NstModuleWrapper:
    if model_name == "vgg19":
        return nst.torch_models.make_vgg19_nst()
    else:
        raise ValueError("unknown model {}".format(model_name))


def _run_nst(model, content_img, style_img, n_iterations,
             content_weight, style_weight, use_gpu):
    if use_gpu:
        device = torch.device("cuda")

    model.to(device)

    with st.spinner("Computing content features ..."):
        content_tensor = _to_tensor(content_img).to(device)
        model.set_content_image(content_tensor)

    with st.spinner("Computing style features ..."):
        style_tensor = _to_tensor(style_img).to(device)
        model.set_style_image(style_tensor)

    input_img_tensor = content_tensor.clone().unsqueeze(0)
    input_img_tensor.requires_grad_()
    optimizer = torch.optim.LBFGS([input_img_tensor])

    pbar = st.progress(0.0)
    for i in range(n_iterations):
        model.run_optimizer_step(input_img_tensor, optimizer,
                                 style_weight, content_weight)
        pbar.progress((i + 1) / n_iterations)


def _to_tensor(pil_image):
    return torchvision.transforms.ToTensor()(pil_image)


def _to_image(tensor):
    return torchvision.transforms.ToPILImage()(tensor)


################################################################################

if __name__ == "__main__":
    main()
